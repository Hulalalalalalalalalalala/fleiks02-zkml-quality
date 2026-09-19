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
    TaskConflict,
    TaskError,
    TaskIdempotencyConflict,
    TaskInvalidState,
    TaskNotFound,
    TaskStoreCorrupt,
    create_task,
    retry_task,
    run_task,
    status_task,
)
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


if __name__ == "__main__":
    unittest.main()
