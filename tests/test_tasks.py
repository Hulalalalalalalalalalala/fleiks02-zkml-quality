"""Tests for the durable offline proof-task store and its CLI.

A real ``zk-setup`` is produced once (EZKL, CPU) because a successful ``run``
must execute the genuine proving loop; tests that need to damage a setup work
on a per-test copy. The proving internals themselves are covered in
``test_zk.py``; here the focus is the task lifecycle, attempts, idempotency,
single-executor claiming, atomic/corrupt-store behaviour and privacy.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from zkml_quality.inference import ROOT
from zkml_quality.tasks import (
    SAFE_MESSAGES,
    TaskAttemptFailed,
    TaskCapacityExceeded,
    TaskConflict,
    TaskError,
    TaskIdempotencyConflict,
    TaskInvalidState,
    TaskNotFound,
    TaskStoreCorrupt,
    batch_create,
    batch_run,
    batch_status,
    create_task,
    retry_task,
    run_task,
    status_task,
)
import zkml_quality.tasks as tasks_mod
from zkml_quality.zk import (
    ZkEzklError,
    ZkOutputError,
    run_setup,
)

MODEL = ROOT / "models" / "quality.onnx"
FEATURES = [0.125] * 6


class ProofTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(tempfile.mkdtemp(prefix="tasktest-"))
        cls.setup_dir = cls.workspace / "setup"
        run_setup(MODEL, cls.setup_dir)
        cls.input_path = cls.workspace / "input.json"
        cls.input_path.write_text(json.dumps({"features": FEATURES}), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=self.workspace, prefix="case-"))
        self.store = self.dir / "store.json"
        self.credential = self.dir / "credential.json"
        self.setup = self.dir / "setup"
        shutil.copytree(self.setup_dir, self.setup)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    # -- helpers ------------------------------------------------------------

    def create(self, *, credential=None, idempotency_key=None,
               input_path=None, setup=None):
        view, created = create_task(
            self.store,
            input_path or self.input_path,
            MODEL,
            setup or self.setup,
            credential or self.credential,
            idempotency_key)
        self.assertTrue(created)
        self.assertEqual(view["state"], "queued")
        self.assertEqual(view["attempt"], 0)
        self.assertRegex(view["task_id"], r"^pt-[0-9a-f]{24}$")
        self.assertIsNone(view["started_at"])
        self.assertIsNone(view["ended_at"])
        self.assertIsNone(view["credential_sha256"])
        return view["task_id"]

    def document(self):
        return json.loads(self.store.read_text(encoding="utf-8"))

    # -- happy path ---------------------------------------------------------

    def test_create_then_real_run_succeeds_with_credential_digest(self):
        tid = self.create()
        view = run_task(self.store, tid)
        self.assertEqual(view["state"], "succeeded")
        self.assertEqual(view["attempt"], 1)
        self.assertIsNotNone(view["started_at"])
        self.assertIsNotNone(view["ended_at"])
        self.assertGreaterEqual(view["ended_at"], view["started_at"])
        self.assertIsNotNone(view["created_at"])
        self.assertEqual(view["credential_sha256"], view["credential_sha256"])
        self.assertEqual(len(view["credential_sha256"]), 64)
        # The recorded digest is the SHA-256 of the real credential file.
        import hashlib
        self.assertEqual(view["credential_sha256"],
                         hashlib.sha256(self.credential.read_bytes()).hexdigest())
        # A genuine zk-prove credential was written.
        credential = json.loads(self.credential.read_text(encoding="utf-8"))
        self.assertEqual(credential["kind"], "zk-quality-credential")
        self.assertGreater(len(credential["proof"]["proof"]), 100)

        record = self.document()["tasks"][tid]
        self.assertEqual(len(record["attempts"]), 1)
        entry = record["attempts"][0]
        self.assertEqual(entry["state"], "succeeded")
        self.assertEqual(entry["attempt"], 1)
        self.assertIsNone(entry["error_code"])
        self.assertIsNone(entry["retryable"])
        self.assertEqual(entry["credential_sha256"], view["credential_sha256"])
        self.assertTrue(entry["started_at"] and entry["ended_at"])

    def test_status_does_not_mutate_store(self):
        tid = self.create()
        before = self.store.read_bytes()
        view = status_task(self.store, tid)
        self.assertEqual(view["state"], "queued")
        self.assertEqual(self.store.read_bytes(), before)

    def test_state_machine_rejects_illegal_transitions(self):
        tid = self.create()
        # queued -> retry is illegal.
        with self.assertRaises(TaskInvalidState):
            retry_task(self.store, tid)
        view = run_task(self.store, tid)
        self.assertEqual(view["state"], "succeeded")
        # succeeded is terminal for both run and retry.
        with self.assertRaises(TaskInvalidState):
            run_task(self.store, tid)
        with self.assertRaises(TaskInvalidState):
            retry_task(self.store, tid)

    def test_unknown_and_malformed_task_ids(self):
        self.create()
        with self.assertRaises(TaskNotFound):
            status_task(self.store, "pt-" + "0" * 24)
        with self.assertRaises(TaskNotFound):
            run_task(self.store, "pt-" + "0" * 24)
        with self.assertRaises(TaskNotFound):
            retry_task(self.store, "pt-" + "0" * 24)
        with self.assertRaises(TaskError) as caught:
            status_task(self.store, "not-an-id")
        self.assertEqual(caught.exception.code, "invalid_request")

    # -- idempotency --------------------------------------------------------

    def test_idempotent_create_returns_original_task(self):
        view1, created1 = create_task(
            self.store, self.input_path, MODEL, self.setup, self.credential, "key-1")
        self.assertTrue(created1)
        view2, created2 = create_task(
            self.store, self.input_path, MODEL, self.setup, self.credential, "key-1")
        self.assertFalse(created2)
        self.assertEqual(view2["task_id"], view1["task_id"])
        self.assertEqual(self.document()["tasks"][view1["task_id"]]["idempotency_key"], "key-1")

    def test_idempotency_conflict_is_rejected_without_writing(self):
        create_task(self.store, self.input_path, MODEL, self.setup,
                    self.credential, "key-1")
        before = self.store.read_bytes()
        other_credential = self.dir / "other.json"
        with self.assertRaises(TaskIdempotencyConflict):
            create_task(self.store, self.input_path, MODEL, self.setup,
                        other_credential, "key-1")
        # Same key, changed input content: conflict too.
        changed_input = self.dir / "changed.json"
        changed_input.write_text(json.dumps({"features": [0.5] * 6}), encoding="utf-8")
        with self.assertRaises(TaskIdempotencyConflict):
            create_task(self.store, changed_input, MODEL, self.setup,
                        self.credential, "key-1")
        self.assertEqual(self.store.read_bytes(), before)

    def test_distinct_requests_without_keys_make_distinct_tasks(self):
        first = self.create()
        second = self.create(credential=self.dir / "two.json")
        self.assertNotEqual(first, second)

    # -- failures, attempts and retry --------------------------------------

    def test_missing_artifact_fails_retryable_then_retry_succeeds(self):
        tid = self.create()
        proving_key = self.setup / "proving.key"
        backup = self.dir / "pk.bak"
        shutil.move(proving_key, backup)
        with self.assertRaises(TaskAttemptFailed) as caught:
            run_task(self.store, tid)
        self.assertEqual(caught.exception.code, "artifact_missing")
        self.assertTrue(caught.exception.retryable)
        failed_view = caught.exception.view
        self.assertEqual(failed_view["state"], "failed")
        self.assertEqual(failed_view["attempt"], 1)
        self.assertIsNone(failed_view["credential_sha256"])
        self.assertFalse(self.credential.exists())

        # Re-queue: attempt counter is preserved, started/ended reset.
        view = retry_task(self.store, tid)
        self.assertEqual(view["state"], "queued")
        self.assertEqual(view["attempt"], 1)
        self.assertIsNone(view["started_at"])
        self.assertIsNone(view["credential_sha256"])

        # Failing again records a second attempt with full history kept.
        with self.assertRaises(TaskAttemptFailed):
            run_task(self.store, tid)
        self.assertEqual(status_task(self.store, tid)["attempt"], 2)
        self.assertEqual(
            [a["state"] for a in self.document()["tasks"][tid]["attempts"]],
            ["failed", "failed"])

        # Repair and retry: attempt 3 performs the real proof.
        retry_task(self.store, tid)
        shutil.move(backup, proving_key)
        view = run_task(self.store, tid)
        self.assertEqual(view["state"], "succeeded")
        self.assertEqual(view["attempt"], 3)
        record = self.document()["tasks"][tid]
        self.assertEqual([a["state"] for a in record["attempts"]],
                         ["failed", "failed", "succeeded"])

    def test_invalid_input_at_create_makes_no_task_and_no_store(self):
        bad_input = self.dir / "bad.json"
        bad_input.write_text(json.dumps({"features": [1] * 5}), encoding="utf-8")
        with self.assertRaises(TaskError) as caught:
            create_task(self.store, bad_input, MODEL, self.setup, self.credential)
        self.assertEqual(caught.exception.code, "invalid_input")
        self.assertFalse(caught.exception.retryable)
        self.assertFalse(self.store.exists())

    def test_missing_credential_output_directory_at_create(self):
        with self.assertRaises(TaskError) as caught:
            create_task(self.store, self.input_path, MODEL, self.setup,
                        self.dir / "no-such-dir" / "credential.json")
        self.assertEqual(caught.exception.code, "artifact_missing")
        self.assertTrue(caught.exception.retryable)
        self.assertFalse(self.store.exists())

    def test_digest_mismatch_after_tampering_is_terminal_failure(self):
        # Input changed between create and run: sealed digest disagrees.
        input_path = self.dir / "sealed.json"
        input_path.write_text(json.dumps({"features": FEATURES}), encoding="utf-8")
        tid = self.create(input_path=input_path)
        input_path.write_text(json.dumps({"features": [0.5] * 6}), encoding="utf-8")
        with self.assertRaises(TaskAttemptFailed) as caught:
            run_task(self.store, tid)
        self.assertEqual(caught.exception.code, "digest_mismatch")
        self.assertFalse(caught.exception.retryable)
        self.assertFalse(self.credential.exists())
        # Non-retryable failure: retry refuses.
        with self.assertRaises(TaskInvalidState):
            retry_task(self.store, tid)

        # A setup artifact whose bytes changed (e.g. swapped between setups)
        # is also a sealed-digest mismatch (tamper after create).
        tid2 = self.create(credential=self.dir / "c2.json")
        vk = self.setup / "verification.key"
        vk.write_bytes(vk.read_bytes() + b"\x00")
        with self.assertRaises(TaskAttemptFailed) as caught2:
            run_task(self.store, tid2)
        self.assertEqual(caught2.exception.code, "digest_mismatch")
        self.assertFalse(caught2.exception.retryable)

    def test_failed_attempt_never_overwrites_existing_credential(self):
        # Pre-place a credential at the target; a failed run must not touch it.
        self.credential.write_bytes(b"pre-existing credential bytes")
        tid = self.create()
        shutil.move(self.setup / "proving.key", self.dir / "pk.bak2")
        with self.assertRaises(TaskAttemptFailed):
            run_task(self.store, tid)
        self.assertEqual(self.credential.read_bytes(), b"pre-existing credential bytes")
        view = status_task(self.store, tid)
        self.assertIsNone(view["credential_sha256"])

    def test_credential_vanishing_after_proof_is_not_reported_success(self):
        tid = self.create()
        import zkml_quality.tasks as tasks_mod
        original = tasks_mod._issue_credential

        def issue_then_delete(features, paths, model_sha, scale, credential_path):
            result = original(features, paths, model_sha, scale, credential_path)
            Path(credential_path).unlink()
            return result

        with mock.patch.object(tasks_mod, "_issue_credential", issue_then_delete):
            with self.assertRaises(TaskAttemptFailed) as caught:
                run_task(self.store, tid)
        self.assertEqual(caught.exception.code, "output_io_failure")
        self.assertTrue(caught.exception.retryable)
        self.assertIsNone(caught.exception.view["credential_sha256"])
        self.assertEqual(caught.exception.view["state"], "failed")

    def test_ezkl_and_output_failures_are_classified(self):
        tid = self.create()
        with mock.patch("zkml_quality.tasks._issue_credential",
                        side_effect=ZkEzklError("boom")):
            with self.assertRaises(TaskAttemptFailed) as caught:
                run_task(self.store, tid)
        self.assertEqual(caught.exception.code, "ezkl_failure")
        self.assertTrue(caught.exception.retryable)
        self.assertFalse(self.credential.exists())

        retry_task(self.store, tid)
        with mock.patch("zkml_quality.tasks._issue_credential",
                        side_effect=ZkOutputError("disk full")):
            with self.assertRaises(TaskAttemptFailed) as caught:
                run_task(self.store, tid)
        self.assertEqual(caught.exception.code, "output_io_failure")
        self.assertTrue(caught.exception.retryable)

    def test_interruption_is_recorded_as_failed_not_success(self):
        tid = self.create()

        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt

        with mock.patch("zkml_quality.tasks._issue_credential", side_effect=interrupt):
            with self.assertRaises(TaskAttemptFailed) as caught:
                run_task(self.store, tid)
        self.assertEqual(caught.exception.code, "interrupted")
        self.assertTrue(caught.exception.retryable)
        view = caught.exception.view
        self.assertEqual(view["state"], "failed")
        self.assertIsNone(view["credential_sha256"])
        # The recorded interrupted attempt can be retried and then succeeds.
        retry_task(self.store, tid)
        view = run_task(self.store, tid)
        self.assertEqual(view["state"], "succeeded")
        self.assertEqual(view["attempt"], 2)

    # -- single executor ----------------------------------------------------

    def test_only_one_executor_can_claim_a_task(self):
        tid = self.create()
        import zkml_quality.tasks as tasks_mod
        entered = threading.Event()
        release = threading.Event()
        original = tasks_mod._issue_credential

        def blocking_issue(*args):
            entered.set()
            self.assertTrue(release.wait(10))
            return original(*args)

        outcome = []

        def worker():
            try:
                run_task(self.store, tid)
                outcome.append("succeeded")
            except BaseException as error:  # pragma: no cover - diagnostic
                outcome.append(getattr(error, "code", repr(error)))

        with mock.patch.object(tasks_mod, "_issue_credential", blocking_issue):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(entered.wait(10))
            # While attempt 1 is genuinely in flight a competing runner loses.
            with self.assertRaises(TaskConflict) as caught:
                run_task(self.store, tid)
            self.assertEqual(caught.exception.code, "conflict")
            release.set()
            thread.join()

        self.assertEqual(outcome, ["succeeded"])
        self.assertEqual(status_task(self.store, tid)["state"], "succeeded")

    # -- store integrity ----------------------------------------------------

    def test_corrupt_or_illegal_store_is_refused_and_preserved(self):
        self.create()
        tid2 = self.create(credential=self.dir / "x.json")

        corrupt_payloads = [
            b"{not json",
            json.dumps(["not", "an", "object"]).encode(),
            json.dumps({"format_version": 2, "kind": "zk-quality-proof-tasks",
                        "tasks": {}}).encode(),
            json.dumps({"format_version": 1, "kind": "other", "tasks": {}}).encode(),
            b'{"format_version": 1, "kind": "zk-quality-proof-tasks",'
            b' "tasks": {}, "tasks": {}}',
        ]
        # Illegal state machine: succeeded history but top-level "queued".
        run_task(self.store, tid2)
        doc = self.document()
        doc["tasks"][tid2]["state"] = "queued"
        corrupt_payloads.append(json.dumps(doc).encode())
        # Broken request seal.
        doc = self.document()
        other_tid = next(iter(doc["tasks"]))
        doc["tasks"][other_tid]["request"]["input_sha256"] = "0" * 64
        corrupt_payloads.append(json.dumps(doc).encode())

        for payload in corrupt_payloads:
            with self.subTest(payload=payload[:32]):
                target = self.dir / "bad-store.json"
                target.write_bytes(payload)
                for operation in (
                        lambda: status_task(target, other_tid),
                        lambda: run_task(target, other_tid),
                        lambda: retry_task(target, other_tid)):
                    with self.assertRaises(TaskStoreCorrupt):
                        operation()
                self.assertEqual(target.read_bytes(), payload)

    def test_successful_updates_leave_no_temp_files(self):
        tid = self.create()
        run_task(self.store, tid)
        leftovers = [p.name for p in self.dir.iterdir()
                     if p.name.startswith(".proof-tasks-")]
        self.assertEqual(leftovers, [])

    # -- privacy ------------------------------------------------------------

    def test_no_features_paths_or_proof_in_views(self):
        tid = self.create()
        queued_view = status_task(self.store, tid)
        succeeded_view = run_task(self.store, tid)
        views_blob = json.dumps([queued_view, succeeded_view])
        for secret in ("features", "input_data", str(self.dir), "proof",
                       "setup", "samples", str(MODEL), "credential.json"):
            self.assertNotIn(secret, views_blob)
        # The on-disk store itself never stores the features or proof; paths
        # are present only because the local worker needs them, and never
        # appear in any command output.
        raw = self.store.read_text(encoding="utf-8")
        self.assertNotIn("features", raw)
        self.assertNotIn("0.125", raw)
        self.assertNotIn('"proof"', raw)

    # -- CLI ----------------------------------------------------------------

    def cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "zkml_quality", "proof-task", *map(str, arguments)],
            cwd=str(ROOT), capture_output=True, text=True)

    def test_cli_success_is_single_json_stdout(self):
        proc = self.cli(
            "create", "--store", self.store, "--input", self.input_path,
            "--model", MODEL, "--setup-dir", self.setup,
            "--credential", self.credential)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        view = json.loads(proc.stdout)
        self.assertEqual(set(view),
                         {"task_id", "state", "attempt", "created_at",
                          "started_at", "ended_at", "updated_at",
                          "credential_sha256"})
        tid = view["task_id"]

        proc = self.cli("run", "--store", self.store, "--task-id", tid)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        result = json.loads(proc.stdout)
        self.assertEqual(result["state"], "succeeded")
        self.assertTrue(result["credential_sha256"])

        proc = self.cli("status", "--store", self.store, "--task-id", tid)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["state"], "succeeded")

    def test_cli_failure_is_stderr_only_nonzero_with_safe_json(self):
        tid = self.create()
        shutil.move(self.setup / "proving.key", self.dir / "pk-cli.bak")
        proc = self.cli("run", "--store", self.store, "--task-id", tid)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        payload = json.loads(proc.stderr)
        self.assertEqual(payload["state"], "failed")
        self.assertEqual(payload["attempt"], 1)
        self.assertEqual(payload["error"]["code"], "artifact_missing")
        self.assertIs(payload["error"]["retryable"], True)
        self.assertEqual(payload["error"]["message"],
                         SAFE_MESSAGES["artifact_missing"])
        for secret in (str(self.dir), "features", "proving.key", str(MODEL)):
            self.assertNotIn(secret, proc.stderr)

        # Invalid request: no task fields, just the error envelope.
        proc = self.cli("status", "--store", self.store, "--task-id", "pt-" + "9" * 24)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        payload = json.loads(proc.stderr)
        self.assertEqual(payload["error"]["code"], "task_not_found")
        self.assertNotIn("task_id", payload)

    def test_cli_idempotency_conflict_and_retry_roundtrip(self):
        proc = self.cli(
            "create", "--store", self.store, "--input", self.input_path,
            "--model", MODEL, "--setup-dir", self.setup,
            "--credential", self.credential, "--idempotency-key", "k")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        tid = json.loads(proc.stdout)["task_id"]

        proc = self.cli(
            "create", "--store", self.store, "--input", self.input_path,
            "--model", MODEL, "--setup-dir", self.setup,
            "--credential", self.dir / "different.json",
            "--idempotency-key", "k")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(json.loads(proc.stderr)["error"]["code"],
                         "idempotency_conflict")

        # Same key/request again returns the same task with exit code 0.
        proc = self.cli(
            "create", "--store", self.store, "--input", self.input_path,
            "--model", MODEL, "--setup-dir", self.setup,
            "--credential", self.credential, "--idempotency-key", "k")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["task_id"], tid)

    # -- canonical paths ----------------------------------------------------

    def test_create_seals_absolute_canonical_paths_and_run_survives_cwd_change(self):
        subdir = self.dir / "nested" / "workdir"
        subdir.mkdir(parents=True)
        (subdir / "out").mkdir()
        (subdir / "input.json").write_text(
            json.dumps({"features": FEATURES}), encoding="utf-8")
        rel_input = Path("input.json")
        rel_model = Path(os.path.relpath(MODEL, subdir))
        rel_setup = Path(os.path.relpath(self.setup, subdir))
        rel_credential = Path("out") / "credential.json"
        store = self.dir / "store.json"
        old_cwd = Path.cwd()
        try:
            os.chdir(subdir)
            view, created = create_task(
                store, rel_input, rel_model, rel_setup, rel_credential)
            self.assertTrue(created)
            tid = view["task_id"]
            doc = json.loads(store.read_text())
            request = doc["tasks"][tid]["request"]
            for key in ("input_path", "model_path", "setup_dir", "credential_path"):
                self.assertTrue(os.path.isabs(request[key]), key)
                self.assertNotIn("..", Path(request[key]).parts)
            self.assertEqual(request["input_path"], str((subdir / "input.json").resolve()))
            self.assertEqual(request["model_path"], str(MODEL.resolve()))
            self.assertEqual(request["setup_dir"], str(self.setup.resolve()))
            # Run from an unrelated directory: the sealed absolute paths must
            # not resolve against the new cwd.
            os.chdir(self.workspace)
            result = run_task(store, tid)
            self.assertEqual(result["state"], "succeeded")
        finally:
            os.chdir(old_cwd)

    # -- retryable store_io -------------------------------------------------

    def test_store_io_failure_is_retryable_and_does_not_lose_the_task(self):
        tid = self.create()
        with mock.patch("zkml_quality.tasks._atomic_write",
                        side_effect=OSError("disk full")):
            with self.assertRaises(TaskError) as caught:
                run_task(self.store, tid)
        self.assertEqual(caught.exception.code, "store_io")
        self.assertTrue(caught.exception.retryable)
        # The claim commit failed, so the task is still queued and unclaimed.
        self.assertEqual(status_task(self.store, tid)["state"], "queued")

    # -- recoverable credential publish -------------------------------------

    def _stage_a_crashed_success(self, tid, *, rename_to_target=False,
                                 drop_staged=False, foreign_target=False):
        """Claim, prove into staging and journal — then stop like a killed run.

        The per-attempt fence is released (as the kernel does on process
        death) while the running+journaled task state is left on disk, so a
        later store open treats it as an interrupted attempt to recover.
        """
        with tasks_mod._locked_store(self.store, create_parents=False,
                                     recover=False) as (document, persist):
            task, attempt_number, fence = tasks_mod._claim(document, tid, self.store)
            request = task["request"]
            persist()
        try:
            ok, info = tasks_mod._execute_attempt(request)
            self.assertTrue(ok, info)
            staged = Path(info["staging_path"])
            target = Path(request["credential_path"])
            if foreign_target:
                target.write_bytes(b"unrelated pre-existing credential")
            if rename_to_target:
                os.replace(staged, target)
            if drop_staged and staged.exists():
                staged.unlink()
            with tasks_mod._locked_store(self.store, create_parents=False,
                                         recover=False) as (document, persist):
                owner = tasks_mod._get_task(document, tid)
                owner["pending_credential"] = {
                    "temp_path": info["staging_path"],
                    "credential_sha256": info["digest"]}
                persist()
        finally:
            # Simulate process death: drop the fence fd without finalising.
            tasks_mod._release_attempt_lock(
                fence, self.store, tid, attempt_number)
        return info["digest"], target, staged

    def _stage_a_crashed_claim(self, tid):
        """Persist a running claim with no journal and a free fence."""
        with tasks_mod._locked_store(self.store, create_parents=False,
                                     recover=False) as (document, persist):
            _task, attempt_number, fence = tasks_mod._claim(document, tid, self.store)
            persist()
        tasks_mod._release_attempt_lock(fence, self.store, tid, attempt_number)
        return attempt_number

    def test_recovery_completes_publish_in_a_fresh_process(self):
        # Killed between journal commit and rename: restart converges success.
        tid = self.create()
        digest, target, staged = self._stage_a_crashed_success(tid)
        self.assertTrue(staged.is_file())
        self.assertFalse(target.exists())
        proc = self.cli("status", "--store", self.store, "--task-id", tid)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        view = json.loads(proc.stdout)
        self.assertEqual(view["state"], "succeeded")
        self.assertEqual(view["credential_sha256"], digest)
        self.assertTrue(target.is_file())
        self.assertFalse(staged.exists())  # staging file consumed
        import hashlib
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest)

    def test_recovery_after_rename_marks_succeeded_without_replacing_file(self):
        tid = self.create()
        digest, target, staged = self._stage_a_crashed_success(
            tid, rename_to_target=True)
        before = target.read_bytes()
        proc = self.cli("status", "--store", self.store, "--task-id", tid)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["state"], "succeeded")
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(staged.exists())

    def test_recovery_with_lost_staging_falls_back_to_retryable_failure(self):
        tid = self.create()
        _digest, target, staged = self._stage_a_crashed_success(
            tid, drop_staged=True)
        proc = self.cli("status", "--store", self.store, "--task-id", tid)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        view = json.loads(proc.stdout)
        self.assertEqual(view["state"], "failed")
        self.assertIsNone(view["credential_sha256"])
        self.assertFalse(target.exists())
        # The recovered failure is the retryable 'interrupted' kind: retry and
        # a real re-run converge to success, consistent after restart.
        record = self.document()["tasks"][tid]
        self.assertEqual(record["last_error"]["code"], "interrupted")
        self.assertTrue(record["last_error"]["retryable"])
        retry_task(self.store, tid)
        self.assertEqual(run_task(self.store, tid)["state"], "succeeded")

    def test_recovery_never_overwrites_a_preexisting_credential(self):
        tid = self.create()
        _digest, target, staged = self._stage_a_crashed_success(
            tid, foreign_target=True)
        original = b"unrelated pre-existing credential"
        proc = self.cli("status", "--store", self.store, "--task-id", tid)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        view = json.loads(proc.stdout)
        self.assertEqual(view["state"], "failed")
        self.assertIsNone(view["credential_sha256"])
        self.assertEqual(target.read_bytes(), original)
        self.assertFalse(staged.exists())

    def test_failed_run_leaves_no_staging_file(self):
        tid = self.create()
        shutil.move(self.setup / "proving.key", self.dir / "pk-leftover.bak")
        with self.assertRaises(TaskAttemptFailed):
            run_task(self.store, tid)
        leftovers = [p.name for p in self.dir.iterdir()
                     if p.name.endswith(".proof-task.tmp")]
        self.assertEqual(leftovers, [])

    def test_recovery_after_death_before_journal_requeues_as_interrupted(self):
        # Killed after the claim commit but before proving/journaling: on the
        # next open the dead owner's fence is free, so the running attempt is
        # reconciled to a retryable interrupted failure (never a stuck running).
        tid = self.create()
        self._stage_a_crashed_claim(tid)
        self.assertEqual(status_task(self.store, tid)["state"], "failed")
        record = self.document()["tasks"][tid]
        self.assertEqual(record["state"], "failed")
        self.assertEqual(record["last_error"]["code"], "interrupted")
        self.assertTrue(record["last_error"]["retryable"])
        self.assertIsNone(record["credential_sha256"])
        retry_task(self.store, tid)
        self.assertEqual(run_task(self.store, tid)["state"], "succeeded")

    def test_recovery_does_not_preempt_a_live_fenced_attempt(self):
        import fcntl
        # Journal a staged publish as if the owning runner is still alive.
        tid = self.create()
        digest, target, staged = self._stage_a_crashed_success(tid)
        doc = self.document()
        attempt_number = doc["tasks"][tid]["attempt"]
        # Simulate the live executor: hold the attempt flock while recovery
        # runs. The task must stay running and no publish may happen.
        lock_path = tasks_mod._attempt_lock_path(self.store, tid, attempt_number)
        live = open(lock_path, "a+b")
        fcntl.flock(live.fileno(), fcntl.LOCK_EX)
        try:
            document = tasks_mod._load_document(self.store, recover=True)
            task = document["tasks"][tid]
            self.assertEqual(task["state"], "running")
            self.assertIsNotNone(task["pending_credential"])
            self.assertFalse(target.exists())
            self.assertTrue(staged.is_file())
            # An ordinary status open likewise leaves it running.
            self.assertEqual(status_task(self.store, tid)["state"], "running")
        finally:
            fcntl.flock(live.fileno(), fcntl.LOCK_UN)
            live.close()
        # Once the live owner exits (lock released), the next open converges.
        view = status_task(self.store, tid)
        self.assertEqual(view["state"], "succeeded")
        self.assertEqual(view["credential_sha256"], digest)


class BatchProofTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(tempfile.mkdtemp(prefix="batchtest-"))
        cls.setup_dir = cls.workspace / "setup"
        run_setup(MODEL, cls.setup_dir)
        cls.input_path = cls.workspace / "input.json"
        cls.input_path.write_text(json.dumps({"features": FEATURES}), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=self.workspace, prefix="case-"))
        self.store = self.dir / "store.json"
        self.setup = self.dir / "setup"
        shutil.copytree(self.setup_dir, self.setup)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def item(self, credential, *, setup=None, input_path=None, key=None):
        raw = {
            "input": str(input_path or self.input_path),
            "model": str(MODEL),
            "setup_dir": str(setup or self.setup),
            "credential": str(credential),
        }
        if key is not None:
            raw["idempotency_key"] = key
        return raw

    def items(self, count, *, keys=False, inputs=None):
        return [self.item(self.dir / f"c{i}.json",
                          key=(f"k-{i}" if keys else None),
                          input_path=(inputs[i] if inputs else None))
                for i in range(count)]

    def test_batch_create_is_all_or_nothing_and_returns_ordered_view(self):
        payload, created = batch_create(self.store, self.items(2))
        self.assertTrue(created)
        self.assertRegex(payload["batch_id"], r"^pb-[0-9a-f]{24}$")
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["counts"],
                         {"queued": 2, "running": 0, "succeeded": 0, "failed": 0})
        self.assertEqual([t["state"] for t in payload["tasks"]], ["queued", "queued"])
        doc = json.loads(self.store.read_text())
        batch_id = payload["batch_id"]
        self.assertIn(batch_id, doc["batches"])
        self.assertEqual(doc["batches"][batch_id]["sequence"], 2)
        for sequence, task in enumerate(
                sorted((t for t in doc["tasks"].values()
                        if t["batch_id"] == batch_id),
                       key=lambda t: t["sequence"])):
            self.assertEqual(task["sequence"], sequence)
            self.assertEqual(task["batch_id"], batch_id)

    def test_batch_create_rejects_whole_batch_when_any_item_invalid(self):
        good = self.items(1)
        bad_input = self.dir / "bad.json"
        bad_input.write_text(json.dumps({"features": [1] * 5}), encoding="utf-8")
        bad = [self.item(self.dir / "cb.json", input_path=bad_input)]
        for order_index, order in enumerate((good + bad, bad + good)):
            store = self.dir / f"store-{order_index}.json"
            with self.assertRaises(TaskError) as caught:
                batch_create(store, order)
            self.assertEqual(caught.exception.code, "invalid_input")
            self.assertFalse(store.exists())

    def test_batch_create_rejects_when_an_artifact_is_missing(self):
        broken_setup = self.dir / "broken-setup"
        shutil.copytree(self.setup, broken_setup)
        (broken_setup / "proving.key").unlink()
        items = self.items(1) + [self.item(self.dir / "cx.json", setup=broken_setup)]
        with self.assertRaises(TaskError) as caught:
            batch_create(self.store, items)
        self.assertEqual(caught.exception.code, "artifact_missing")
        self.assertFalse(self.store.exists())

    def test_batch_create_enforces_queued_capacity_store_wide(self):
        # One standalone queued task already occupies the store.
        create_task(self.store, self.input_path, MODEL, self.setup,
                    self.dir / "solo.json")
        with self.assertRaises(TaskCapacityExceeded) as caught:
            batch_create(self.store, self.items(2), max_queued=2)
        self.assertEqual(caught.exception.code, "capacity_exceeded")
        self.assertFalse(caught.exception.retryable)
        before = self.store.read_bytes()
        # Exactly at the limit boundary is allowed; over is not.
        payload, created = batch_create(self.store, self.items(1), max_queued=2)
        self.assertTrue(created)
        self.assertEqual(payload["counts"]["queued"], 1)
        self.assertNotEqual(self.store.read_bytes(), before)

    def test_batch_create_idempotent_replay_returns_stable_batch_id(self):
        first, created1 = batch_create(self.store, self.items(2, keys=True))
        self.assertTrue(created1)
        second, created2 = batch_create(self.store, self.items(2, keys=True))
        self.assertFalse(created2)
        self.assertEqual(second["batch_id"], first["batch_id"])
        self.assertEqual([t["task_id"] for t in second["tasks"]],
                         [t["task_id"] for t in first["tasks"]])
        # Same keys, one request changed: conflict, no write.
        changed = self.items(2, keys=True)
        changed[1]["credential"] = str(self.dir / "different.json")
        with self.assertRaises(TaskIdempotencyConflict):
            batch_create(self.store, changed)
        # A key repeated inside the file is itself a conflict.
        dup = self.items(2)
        dup[0]["idempotency_key"] = dup[1]["idempotency_key"] = "same"
        with self.assertRaises(TaskIdempotencyConflict):
            batch_create(self.dir / "dup.json", dup)

    def test_batch_status_unknown_batch(self):
        batch_create(self.store, self.items(1))
        with self.assertRaises(TaskError) as caught:
            batch_status(self.store, "pb-" + "0" * 24)
        self.assertEqual(caught.exception.code, "batch_not_found")

    def test_batch_run_really_proves_every_item_with_bounded_workers(self):
        payload, _ = batch_create(self.store, self.items(3, keys=False))
        result = batch_run(self.store, payload["batch_id"], 2)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["counts"],
                         {"queued": 0, "running": 0, "succeeded": 3, "failed": 0})
        for view in result["tasks"]:
            self.assertEqual(view["state"], "succeeded")
            self.assertEqual(view["attempt"], 1)
            self.assertTrue(view["credential_sha256"])
        # Creation order is preserved in the reported views.
        self.assertEqual([t["task_id"] for t in result["tasks"]],
                         [t["task_id"] for t in payload["tasks"]])
        for i in range(3):
            self.assertTrue((self.dir / f"c{i}.json").is_file())

    def test_batch_run_one_failure_does_not_block_others_and_task_is_retryable(self):
        broken_setup = self.dir / "setup-broken"
        shutil.copytree(self.setup, broken_setup)
        items = [
            self.item(self.dir / "ok.json"),
            self.item(self.dir / "broken.json", setup=broken_setup),
        ]
        payload, _ = batch_create(self.store, items)
        batch_id = payload["batch_id"]
        (broken_setup / "proving.key").unlink()
        result = batch_run(self.store, batch_id, 2)
        self.assertEqual(result["counts"]["succeeded"], 1)
        self.assertEqual(result["counts"]["failed"], 1)
        failed_view = next(t for t in result["tasks"] if t["state"] == "failed")
        self.assertEqual(failed_view["attempt"], 1)
        # Ordinary single-task retry applies to a batch member.
        retry_task(self.store, failed_view["task_id"])
        shutil.copy2(self.setup / "proving.key", broken_setup / "proving.key")
        view = run_task(self.store, failed_view["task_id"])
        self.assertEqual(view["state"], "succeeded")
        final = batch_status(self.store, batch_id)
        self.assertEqual(final["counts"]["succeeded"], 2)

    def test_batch_run_only_claims_its_own_batch(self):
        # Two batches share one store; running A must not touch B.
        a, _ = batch_create(self.store, self.items(2))
        b_items = [self.item(self.dir / "b0.json")]
        b, _ = batch_create(self.store, b_items)
        result = batch_run(self.store, a["batch_id"], 2)
        self.assertEqual(result["counts"]["succeeded"], 2)
        untouched = batch_status(self.store, b["batch_id"])
        self.assertEqual(untouched["counts"]["queued"], 1)
        self.assertEqual(untouched["tasks"][0]["task_id"],
                         b["tasks"][0]["task_id"])

    def test_batch_run_requires_positive_max_workers(self):
        payload, _ = batch_create(self.store, self.items(1))
        for bad in (0, -1, -16):
            with self.assertRaises(TaskError) as caught:
                batch_run(self.store, payload["batch_id"], bad)
            self.assertEqual(caught.exception.code, "invalid_request")

    # -- batch CLI ----------------------------------------------------------

    def cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "zkml_quality", "proof-task", *map(str, arguments)],
            cwd=str(ROOT), capture_output=True, text=True)

    def write_batch_file(self, path, items):
        path.write_text(json.dumps({"items": items}), encoding="utf-8")

    def test_cli_batch_roundtrip_and_safe_output(self):
        batch_file = self.dir / "batch.json"
        self.write_batch_file(batch_file, self.items(2))
        proc = self.cli("batch-create", "--store", self.store, "--file", batch_file)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        created = json.loads(proc.stdout)
        self.assertEqual(set(created), {"batch_id", "total", "counts", "tasks"})
        batch_id = created["batch_id"]

        proc = self.cli("batch-run", "--store", self.store,
                        "--batch-id", batch_id, "--max-workers", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        result = json.loads(proc.stdout)
        self.assertEqual(result["counts"]["succeeded"], 2)
        for blob in (created, result):
            text = json.dumps(blob)
            for secret in ("features", "0.125", "proof", "setup_dir",
                           "input_path", str(self.dir), str(MODEL), ".onnx"):
                self.assertNotIn(secret, text)

        proc = self.cli("batch-status", "--store", self.store,
                        "--batch-id", batch_id)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["total"], 2)

    def test_cli_batch_failures_are_stderr_only_safe_json(self):
        # Capacity exceeded.
        batch_file = self.dir / "batch.json"
        self.write_batch_file(batch_file, self.items(2))
        proc = self.cli("batch-create", "--store", self.store,
                        "--file", batch_file, "--max-queued", "1")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        payload = json.loads(proc.stderr)
        self.assertEqual(payload["error"]["code"], "capacity_exceeded")
        self.assertIs(payload["error"]["retryable"], False)
        self.assertNotIn(str(self.dir), proc.stderr)

        # Non-positive --max-workers is an invalid request.
        created, _created_flag = batch_create(self.store, self.items(1))
        proc = self.cli("batch-run", "--store", self.store,
                        "--batch-id", created["batch_id"], "--max-workers", "0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(json.loads(proc.stderr)["error"]["code"], "invalid_request")

        # Unknown batch.
        proc = self.cli("batch-status", "--store", self.store,
                        "--batch-id", "pb-" + "0" * 24)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertEqual(json.loads(proc.stderr)["error"]["code"], "batch_not_found")


if __name__ == "__main__":
    unittest.main()
