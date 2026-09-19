"""End-to-end tests for the offline proof-task queue (proof-task)."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from zkml_quality.inference import ROOT
from zkml_quality.prooftask import (
    ERR_ARTIFACT_MISSING,
    ERR_DIGEST_MISMATCH,
    ERR_INTERRUPTED,
    ERR_INVALID_REQUEST,
    ERR_STORE_CORRUPT,
    ProofTaskError,
    create_task,
    retry_task,
    run_task,
    status_task,
)
from zkml_quality.registry import enable_model, register_model
from zkml_quality.zk import run_setup, run_verify

MODEL = ROOT / "models" / "quality.onnx"
SAMPLE = ROOT / "samples" / "normal.json"
VERSION = "v1.0"

OUTPUT_KEYS = {"task_id", "state", "attempt", "created_at", "started_at",
               "finished_at", "credential_sha256"}


def features_document(values):
    return {"features": list(values)}


class ProofTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(tempfile.mkdtemp(prefix="zkprooftask-"))
        cls.setup_dir = cls.workspace / "setup"
        run_setup(MODEL, cls.setup_dir)
        cls.store = cls.workspace / "store"
        cls.input_path = cls.workspace / "input.json"
        cls.input_path.write_text(json.dumps(features_document([0.125] * 6)),
                                  encoding="utf-8")

        # Verifier side, to prove credentials issued by tasks stay compatible.
        cls.verifier_dir = cls.workspace / "verifier"
        cls.verifier_dir.mkdir()
        shutil.copy(cls.setup_dir / "manifest.json", cls.verifier_dir / "manifest.json")
        shutil.copy(cls.setup_dir / "settings.json", cls.verifier_dir / "settings.json")
        shutil.copy(cls.setup_dir / "verification.key", cls.verifier_dir / "vk")
        shutil.copy(cls.setup_dir / "srs", cls.verifier_dir / "srs")
        shutil.copy(MODEL, cls.verifier_dir / "model.onnx")
        cls.registry = cls.workspace / "registry.json"
        register_model(cls.registry, VERSION,
                       cls.verifier_dir / "manifest.json", cls.verifier_dir / "model.onnx")
        enable_model(cls.registry, VERSION)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _create(self, name="task", key=None, credential=None, input_path=None):
        credential = credential or self.workspace / f"{name}.cred.json"
        return create_task(self.store, input_path or self.input_path, MODEL,
                           self.setup_dir, credential, key)

    def _credential_path(self, task_id, name="task"):
        return self.workspace / f"{name}.cred.json"

    def test_create_shape_and_privacy(self):
        result = self._create("create")
        self.assertEqual(set(result), OUTPUT_KEYS)
        self.assertEqual(result["state"], "queued")
        self.assertEqual(result["attempt"], 0)
        self.assertIsNone(result["started_at"])
        self.assertIsNone(result["finished_at"])
        self.assertIsNone(result["credential_sha256"])
        self.assertTrue(result["task_id"].startswith("pt-"))
        serialized = json.dumps(result)
        for forbidden in (str(self.workspace), "features", "input", "proof",
                          str(self.setup_dir), "0.125"):
            self.assertNotIn(forbidden, serialized)

    def test_create_idempotency(self):
        first = self._create("idem", key="key-1")
        again = self._create("idem", key="key-1")
        self.assertEqual(first["task_id"], again["task_id"])
        # A different request under the same key is a conflict.
        with self.assertRaises(ProofTaskError) as ctx:
            self._create("idem-other", key="key-1")
        self.assertEqual(ctx.exception.code, ERR_INVALID_REQUEST)
        # Without a key every create is a fresh task.
        self.assertNotEqual(self._create("fresh-1")["task_id"],
                            self._create("fresh-2")["task_id"])

    def test_create_rejects_invalid_request(self):
        bad_input = self.workspace / "bad.json"
        bad_input.write_text(json.dumps({"features": [1, 2, 3]}), encoding="utf-8")
        for kwargs in (
                {"input_path": bad_input},
                {"input_path": self.workspace / "missing.json"},
        ):
            with self.assertRaises(ProofTaskError) as ctx:
                self._create("invalid", **kwargs)
            self.assertEqual(ctx.exception.code, ERR_INVALID_REQUEST)
        with self.assertRaises(ProofTaskError):
            self._create("invalid-key", key="")

    def test_run_success_and_verify_compatibility(self):
        result = self._create("run")
        done = run_task(self.store, result["task_id"])
        self.assertEqual(done["state"], "succeeded")
        self.assertEqual(done["attempt"], 1)
        self.assertIsNotNone(done["started_at"])
        self.assertIsNotNone(done["finished_at"])
        credential_path = self._credential_path(done["task_id"], "run")
        self.assertTrue(credential_path.is_file())
        import hashlib
        self.assertEqual(done["credential_sha256"],
                         hashlib.sha256(credential_path.read_bytes()).hexdigest())
        # The credential a task issues passes the unchanged zk-verify loop.
        verified = run_verify(
            credential_path, self.verifier_dir / "manifest.json",
            self.verifier_dir / "model.onnx", self.verifier_dir / "settings.json",
            self.verifier_dir / "vk", self.verifier_dir / "srs",
            self.registry, VERSION)
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["label"], "normal")

    def test_run_rejects_non_queued_states(self):
        result = self._create("states")
        run_task(self.store, result["task_id"])
        with self.assertRaises(ProofTaskError) as ctx:
            run_task(self.store, result["task_id"])
        self.assertEqual(ctx.exception.code, ERR_INVALID_REQUEST)
        with self.assertRaises(ProofTaskError):
            retry_task(self.store, result["task_id"])

    def test_missing_artifact_is_retryable_and_recovers(self):
        result = self._create("missing")
        moved = self.workspace / "input.hidden"
        self.input_path.rename(moved)
        try:
            with self.assertRaises(ProofTaskError) as ctx:
                run_task(self.store, result["task_id"])
            self.assertEqual(ctx.exception.code, ERR_ARTIFACT_MISSING)
            self.assertTrue(ctx.exception.retryable)
            status = status_task(self.store, result["task_id"])
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["error"]["code"], ERR_ARTIFACT_MISSING)
            self.assertIsNone(status["credential_sha256"])
            retried = retry_task(self.store, result["task_id"])
            self.assertEqual(retried["state"], "queued")
            self.assertEqual(retried["attempt"], 1)
        finally:
            moved.rename(self.input_path)
        done = run_task(self.store, result["task_id"])
        self.assertEqual(done["state"], "succeeded")
        self.assertEqual(done["attempt"], 2)

    def test_digest_mismatch_is_not_retryable(self):
        tampered = self.workspace / "tampered.json"
        tampered.write_text(json.dumps(features_document([0.75] * 6)), encoding="utf-8")
        result = self._create("tamper", input_path=tampered)
        tampered.write_text(json.dumps(features_document([0.5] * 6)), encoding="utf-8")
        with self.assertRaises(ProofTaskError) as ctx:
            run_task(self.store, result["task_id"])
        self.assertEqual(ctx.exception.code, ERR_DIGEST_MISMATCH)
        self.assertFalse(ctx.exception.retryable)
        with self.assertRaises(ProofTaskError) as ctx:
            retry_task(self.store, result["task_id"])
        self.assertEqual(ctx.exception.code, ERR_INVALID_REQUEST)

    def test_interrupted_run_records_failure_without_credential(self):
        result = self._create("interrupt")
        credential_path = self._credential_path(result["task_id"], "interrupt")
        with mock.patch("zkml_quality.prooftask.run_prove",
                        side_effect=KeyboardInterrupt):
            with self.assertRaises(ProofTaskError) as ctx:
                run_task(self.store, result["task_id"])
        self.assertEqual(ctx.exception.code, ERR_INTERRUPTED)
        self.assertTrue(ctx.exception.retryable)
        self.assertFalse(credential_path.exists())
        status = status_task(self.store, result["task_id"])
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["error"]["code"], ERR_INTERRUPTED)
        # History survives and the task can be re-queued.
        record = json.loads(
            (self.store / "tasks" / f"{result['task_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(len(record["history"]), 1)
        self.assertEqual(record["history"][0]["error"]["code"], ERR_INTERRUPTED)
        self.assertEqual(retry_task(self.store, result["task_id"])["state"], "queued")

    def test_failed_run_never_overwrites_existing_credential(self):
        credential_path = self.workspace / "occupied.cred.json"
        credential_path.write_text("{}", encoding="utf-8")
        before = credential_path.read_bytes()
        result = self._create("occupied", credential=credential_path)
        with mock.patch("zkml_quality.prooftask.run_prove",
                        side_effect=RuntimeError("boom")):
            with self.assertRaises(ProofTaskError):
                run_task(self.store, result["task_id"])
        self.assertEqual(credential_path.read_bytes(), before)
        self.assertIsNone(status_task(self.store, result["task_id"])["credential_sha256"])

    def test_competing_executor_is_rejected(self):
        result = self._create("locked")
        locks = self.store / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        (locks / f"{result['task_id']}.lock").write_text("pid 1\n", encoding="utf-8")
        with self.assertRaises(ProofTaskError) as ctx:
            run_task(self.store, result["task_id"])
        self.assertEqual(ctx.exception.code, ERR_INVALID_REQUEST)
        self.assertEqual(status_task(self.store, result["task_id"])["state"], "queued")

    def test_corrupt_store_is_rejected_and_left_untouched(self):
        result = self._create("corrupt")
        task_file = self.store / "tasks" / f"{result['task_id']}.json"
        task_file.write_text("not json", encoding="utf-8")
        before = task_file.read_bytes()
        for action in (status_task, run_task, retry_task):
            with self.assertRaises(ProofTaskError) as ctx:
                action(self.store, result["task_id"])
            self.assertEqual(ctx.exception.code, ERR_STORE_CORRUPT)
        self.assertEqual(task_file.read_bytes(), before)

    def test_unknown_or_malformed_task_id(self):
        with self.assertRaises(ProofTaskError) as ctx:
            status_task(self.store, "../escape")
        self.assertEqual(ctx.exception.code, ERR_INVALID_REQUEST)
        with self.assertRaises(ProofTaskError) as ctx:
            status_task(self.store, "pt-" + "0" * 32)
        self.assertEqual(ctx.exception.code, ERR_INVALID_REQUEST)


class ProofTaskCliTest(unittest.TestCase):
    """CLI contract: success prints one JSON object; failure writes stderr only."""

    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(tempfile.mkdtemp(prefix="zkprooftask-cli-"))
        cls.setup_dir = cls.workspace / "setup"
        run_setup(MODEL, cls.setup_dir)
        cls.store = cls.workspace / "store"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _cli(self, *argv):
        return subprocess.run(
            [sys.executable, "-m", "zkml_quality", "proof-task", *argv],
            cwd=ROOT, capture_output=True, text=True)

    def _create_cli(self, name):
        return self._cli(
            "create", "--store", str(self.store),
            "--input", str(ROOT / "samples" / "normal.json"),
            "--model", str(MODEL), "--setup-dir", str(self.setup_dir),
            "--credential", str(self.workspace / f"{name}.cred.json"))

    def test_create_and_status_print_single_json(self):
        created = self._create_cli("cli")
        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertEqual(created.stderr, "")
        result = json.loads(created.stdout)
        self.assertEqual(set(result), OUTPUT_KEYS)
        self.assertNotIn(str(self.workspace), created.stdout)

        status = self._cli("status", "--store", str(self.store),
                           "--task-id", result["task_id"])
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)["state"], "queued")

    def test_run_failure_writes_stderr_only(self):
        created = json.loads(self._create_cli("cli-fail").stdout)
        missing_store = self.workspace / "no-such-store"
        failed = self._cli("run", "--store", str(missing_store),
                           "--task-id", created["task_id"])
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(failed.stdout, "")
        self.assertTrue(failed.stderr.startswith("error: "))
        self.assertNotIn(str(self.workspace), failed.stderr)

        unknown = self._cli("run", "--store", str(self.store),
                            "--task-id", created["task_id"][:-1] + "0")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertEqual(unknown.stdout, "")

    def test_run_success_cli(self):
        created = json.loads(self._create_cli("cli-run").stdout)
        done = self._cli("run", "--store", str(self.store),
                         "--task-id", created["task_id"])
        self.assertEqual(done.returncode, 0, done.stderr)
        result = json.loads(done.stdout)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["attempt"], 1)
        self.assertIsNotNone(result["credential_sha256"])
        # A succeeded task cannot run again.
        again = self._cli("run", "--store", str(self.store),
                          "--task-id", created["task_id"])
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(again.stdout, "")


if __name__ == "__main__":
    unittest.main()
