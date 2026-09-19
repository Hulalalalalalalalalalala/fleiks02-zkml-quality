"""Tests for the offline model registry (no EZKL proving loop needed)."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from zkml_quality.inference import ROOT
from zkml_quality.registry import (
    enable_model,
    list_models,
    register_model,
    revoke_model,
)
from zkml_quality.zk import ZkError, ezkl_version


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="zkregistry-"))
        self.addCleanup(shutil.rmtree, self.workspace, True)
        self.registry = self.workspace / "registry.json"
        self.model = self.workspace / "model.onnx"
        self.model.write_bytes(b"fake onnx bytes")
        self.manifest = self._write_manifest(self.model)

    def _write_manifest(self, model_path, name="manifest.json", **overrides):
        document = {
            "format_version": 1,
            "kind": "zk-quality-setup",
            "ezkl_version": ezkl_version(),
            "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            "logrows": 17,
            "output_scale": 13,
            "artifacts": {key: f"{index:064x}" for index, key in
                          enumerate(("compiled", "settings", "pk", "vk", "srs"))},
        }
        document.update(overrides)
        path = self.workspace / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _read_registry(self):
        return json.loads(self.registry.read_text(encoding="utf-8"))

    def test_register_creates_registry_disabled_and_is_idempotent(self):
        self.assertFalse(self.registry.exists())
        outcome = register_model(self.registry, "1.0.0", self.manifest, self.model)
        self.assertEqual(outcome, {"version": "1.0.0", "status": "disabled", "changed": True})
        record = self._read_registry()["models"]["1.0.0"]
        self.assertEqual(record["status"], "disabled")
        self.assertEqual(record["model_sha256"],
                         hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(record["manifest_sha256"],
                         hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(record["ezkl_version"], ezkl_version())
        self.assertEqual(record["output_scale"], 13)
        self.assertEqual(set(record["artifacts"]), {"settings", "vk", "srs"})

        # Same version, same content: idempotent no-op, no rewrite.
        before = self.registry.read_bytes()
        outcome = register_model(self.registry, "1.0.0", self.manifest, self.model)
        self.assertEqual(outcome["changed"], False)
        self.assertEqual(outcome["status"], "disabled")
        self.assertEqual(self.registry.read_bytes(), before)

    def test_register_refuses_conflicting_content(self):
        register_model(self.registry, "1.0.0", self.manifest, self.model)
        other_model = self.workspace / "other.onnx"
        other_model.write_bytes(b"different bytes")
        other_manifest = self._write_manifest(other_model, name="other-manifest.json")
        with self.assertRaises(ZkError):
            register_model(self.registry, "1.0.0", other_manifest, other_model)
        # The original record is untouched.
        self.assertEqual(self._read_registry()["models"]["1.0.0"]["model_sha256"],
                         hashlib.sha256(self.model.read_bytes()).hexdigest())

    def test_register_validates_version_manifest_and_model(self):
        for bad_version in ("", "a b", "a/b", "v1.0!", "版本", "a:b"):
            with self.subTest(bad_version=bad_version), self.assertRaises(ZkError):
                register_model(self.registry, bad_version, self.manifest, self.model)
        self.assertFalse(self.registry.exists())

        # Model that does not match the manifest's pinned digest.
        other_model = self.workspace / "other.onnx"
        other_model.write_bytes(b"other")
        with self.assertRaises(ZkError):
            register_model(self.registry, "1.0.0", self.manifest, other_model)
        # Malformed manifest.
        broken = self.workspace / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ZkError):
            register_model(self.registry, "1.0.0", broken, self.model)
        # Missing files.
        with self.assertRaises(ZkError):
            register_model(self.registry, "1.0.0", self.workspace / "none.json", self.model)
        with self.assertRaises(ZkError):
            register_model(self.registry, "1.0.0", self.manifest, self.workspace / "none.onnx")
        self.assertFalse(self.registry.exists())

    def test_list_is_lexicographic_by_version(self):
        self.assertEqual(list_models(self.registry), [])  # missing registry lists empty
        for version in ("10.0", "2.0", "1.0", "a-1", "A.1"):
            register_model(self.registry, version, self.manifest, self.model)
        versions = [entry["version"] for entry in list_models(self.registry)]
        self.assertEqual(versions, sorted(versions))
        self.assertEqual(versions, ["1.0", "10.0", "2.0", "A.1", "a-1"])
        for entry in list_models(self.registry):
            self.assertEqual(entry["status"], "disabled")

    def test_enable_and_revoke_lifecycle(self):
        register_model(self.registry, "1.0.0", self.manifest, self.model)
        self.assertEqual(enable_model(self.registry, "1.0.0")["status"], "enabled")
        # Repeated enable is idempotent.
        self.assertEqual(enable_model(self.registry, "1.0.0")["status"], "enabled")
        self.assertEqual(self._read_registry()["models"]["1.0.0"]["status"], "enabled")

        self.assertEqual(revoke_model(self.registry, "1.0.0")["status"], "revoked")
        # Repeated revoke is idempotent.
        self.assertEqual(revoke_model(self.registry, "1.0.0")["status"], "revoked")
        # Revoked is irreversible.
        with self.assertRaises(ZkError):
            enable_model(self.registry, "1.0.0")
        self.assertEqual(self._read_registry()["models"]["1.0.0"]["status"], "revoked")
        # A revoked record cannot be overwritten by conflicting content.
        other_model = self.workspace / "other.onnx"
        other_model.write_bytes(b"other")
        other_manifest = self._write_manifest(other_model, name="other-manifest.json")
        with self.assertRaises(ZkError):
            register_model(self.registry, "1.0.0", other_manifest, other_model)

        # Unknown versions cannot be enabled or revoked.
        with self.assertRaises(ZkError):
            enable_model(self.registry, "9.9.9")
        with self.assertRaises(ZkError):
            revoke_model(self.registry, "9.9.9")
        with self.assertRaises(ZkError):
            enable_model(self.workspace / "missing-registry.json", "1.0.0")

    def test_corrupt_or_invalid_registry_is_rejected_and_preserved(self):
        register_model(self.registry, "1.0.0", self.manifest, self.model)

        def assert_refused_and_preserved(payload):
            self.registry.write_text(payload, encoding="utf-8")
            for operation in (
                    lambda: list_models(self.registry),
                    lambda: register_model(self.registry, "2.0.0", self.manifest, self.model),
                    lambda: enable_model(self.registry, "1.0.0"),
                    lambda: revoke_model(self.registry, "1.0.0")):
                with self.assertRaises(ZkError):
                    operation()
            self.assertEqual(self.registry.read_text(encoding="utf-8"), payload)

        assert_refused_and_preserved("{not json")
        assert_refused_and_preserved('["not", "an", "object"]')
        assert_refused_and_preserved(json.dumps({"format_version": 2,
                                                 "kind": "zk-quality-model-registry",
                                                 "models": {}}))
        assert_refused_and_preserved(json.dumps({
            "format_version": 1, "kind": "zk-quality-model-registry",
            "models": {"1.0.0": {"status": "enabled"}}}))
        assert_refused_and_preserved(json.dumps({
            "format_version": 1, "kind": "zk-quality-model-registry",
            "models": {"bad version!": {}}}))

    def test_cli_roundtrip(self):
        def run_cli(*arguments):
            return subprocess.run(
                [sys.executable, "-m", "zkml_quality", "model-registry", *arguments],
                cwd=str(ROOT), capture_output=True, text=True)

        proc = run_cli("register", "--registry", str(self.registry),
                       "--version", "1.0.0", "--manifest", str(self.manifest),
                       "--model", str(self.model))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "disabled")

        proc = run_cli("enable", "--registry", str(self.registry), "--version", "1.0.0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "enabled")

        proc = run_cli("list", "--registry", str(self.registry))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([entry["version"] for entry in json.loads(proc.stdout)["models"]],
                         ["1.0.0"])

        proc = run_cli("revoke", "--registry", str(self.registry), "--version", "1.0.0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "revoked")

        # Failures exit nonzero, print nothing to stdout and keep the file.
        before = self.registry.read_bytes()
        proc = run_cli("enable", "--registry", str(self.registry), "--version", "1.0.0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("error:", proc.stderr)
        self.assertEqual(self.registry.read_bytes(), before)

        proc = run_cli("register", "--registry", str(self.registry), "--version", "3.0.0")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("--manifest", proc.stderr)


if __name__ == "__main__":
    unittest.main()
