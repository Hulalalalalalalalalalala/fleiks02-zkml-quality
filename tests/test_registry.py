"""Tests for the local offline model registry.

These tests never invoke EZKL: a setup manifest is a pinned-digest document,
so a structurally valid manifest synthesised around the bundled ONNX model is
sufficient to exercise registration, the lifecycle, conflict handling and
on-disk behaviour. The integration with real ``zk-verify`` material is covered
in ``test_zk.py``.
"""
import copy
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
    RegistryError,
    admit_version,
    enable_model,
    list_models,
    load_registry,
    register_model,
    require_record_matches,
    revoke_model,
)
from zkml_quality.zk import ezkl_version

MODEL = ROOT / "models" / "quality.onnx"


def _digest(i):
    return f"{i:0{64}x}"[:64]


def make_manifest(model_sha=None, *, vk_nonce=1, output_scale=13,
                  ezkl=None, corrupt_artifacts=False):
    if model_sha is None:
        model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    artifacts = {
        "compiled": _digest(0x10 + vk_nonce),
        "settings": _digest(0x20 + vk_nonce),
        "pk": _digest(0x30 + vk_nonce),
        "vk": _digest(0x40 + vk_nonce),
        "srs": _digest(0x50 + vk_nonce),
    }
    if corrupt_artifacts:
        artifacts["vk"] = "nope"
    return {
        "format_version": 1,
        "kind": "zk-quality-setup",
        "ezkl_version": ezkl or ezkl_version(),
        "model_sha256": model_sha,
        "logrows": 17,
        "output_scale": output_scale,
        "artifacts": artifacts,
    }


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="regtest-"))
        self.model = self.workspace / "model.onnx"
        shutil.copy(MODEL, self.model)
        self.manifest_path = self.workspace / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(make_manifest(), indent=2, sort_keys=True), encoding="utf-8")
        self.registry = self.workspace / "registry.json"

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _write_manifest(self, name, manifest):
        path = self.workspace / name
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def _register(self, version="v1", manifest=None, model=None):
        return register_model(
            self.registry, version,
            self.manifest_path if manifest is None else manifest,
            self.model if model is None else model)

    # -- creation, validation and initial disabled state -------------------

    def test_registry_file_is_created_and_version_starts_disabled(self):
        self.assertFalse(self.registry.exists())
        nested = self.workspace / "nested" / "dir" / "registry.json"
        result = register_model(nested, "v1", self.manifest_path, self.model)
        self.assertTrue(nested.is_file())
        self.assertEqual(result, {"version": "v1", "status": "disabled", "changed": True})
        document = load_registry(nested)
        self.assertEqual(document["format_version"], 1)
        self.assertEqual(document["kind"], "model-registry")
        self.assertEqual(document["models"]["v1"]["status"], "disabled")
        self.assertEqual(list(document), ["format_version", "kind", "models"])

    def test_version_must_match_the_allowed_token_pattern(self):
        for bad in ("", "v 1", "v/1", "v:1", "v#1", "héllo", "v\n1", 7, None):
            with self.assertRaises(RegistryError):
                self._register(version=bad)
        for good in ("v1", "1.0.0", "model_2026-09", "A..Z", "x-y_z.1"):
            path = self.workspace / f"reg-{good.replace('.', '_')}.json"
            self.assertEqual(
                register_model(path, good, self.manifest_path, self.model)["status"],
                "disabled")

    def test_register_validates_manifest_and_model(self):
        with self.assertRaises(RegistryError):
            register_model(self.registry, "v1", self.workspace / "missing.json", self.model)
        with self.assertRaises(RegistryError):
            register_model(self.registry, "v1", self.manifest_path,
                           self.workspace / "missing.onnx")
        # Manifest the model does not match.
        other_sha = _digest(0xABC)
        bad_manifest = self._write_manifest("m-wrong.json", make_manifest(other_sha))
        with self.assertRaises(RegistryError):
            register_model(self.registry, "v1", bad_manifest, self.model)
        # Malformed manifest fields (bad EZKL version, bad digest, bad scale).
        for kwargs, field in (
                ({"ezkl": "9.9.9"}, "ezkl"),
                ({"corrupt_artifacts": True}, "digest"),
                ({"output_scale": 0}, "scale")):
            bad = self._write_manifest(f"m-{field}.json", make_manifest(**kwargs))
            with self.assertRaises(RegistryError):
                register_model(self.registry, "v1", bad, self.model)
        # Nothing was ever written.
        self.assertFalse(self.registry.exists())

    # -- idempotency and conflicts -----------------------------------------

    def test_same_version_same_content_is_idempotent_and_keeps_status(self):
        first = self._register("v1")
        second = self._register("v1")
        self.assertEqual(first["changed"], True)
        self.assertEqual(second, {"version": "v1", "status": "disabled", "changed": False})

        enable_model(self.registry, "v1")
        again = self._register("v1")
        self.assertEqual(again, {"version": "v1", "status": "enabled", "changed": False})
        self.assertEqual(admit_version(self.registry, "v1")["status"], "enabled")

    def test_conflicting_content_is_rejected_and_never_overwrites(self):
        self._register("v1")
        before = self.registry.read_bytes()

        changed = make_manifest(vk_nonce=2)  # same model, different VK digest
        changed_manifest = self._write_manifest("m2.json", changed)
        with self.assertRaises(RegistryError):
            register_model(self.registry, "v1", changed_manifest, self.model)
        self.assertEqual(self.registry.read_bytes(), before)

        # A different version with the new content is fine.
        other = register_model(self.registry, "v2", changed_manifest, self.model)
        self.assertEqual(other["status"], "disabled")
        entries = {item["version"]: item for item in list_models(self.registry)}
        self.assertNotEqual(entries["v1"]["artifacts"]["vk"], entries["v2"]["artifacts"]["vk"])

    def test_conflict_after_revoke_still_rejected(self):
        self._register("v1")
        revoke_model(self.registry, "v1")
        changed = self._write_manifest("m3.json", make_manifest(vk_nonce=3))
        with self.assertRaises(RegistryError):
            register_model(self.registry, "v1", changed, self.model)
        # Identical content remains idempotent and stays revoked.
        result = self._register("v1")
        self.assertEqual(result, {"version": "v1", "status": "revoked", "changed": False})
        with self.assertRaises(RegistryError):
            admit_version(self.registry, "v1")

    # -- list ordering ------------------------------------------------------

    def test_list_is_lexicographic_by_version(self):
        for version in ("v10", "v2", "v1", "2026.01", "2026.1", "a"):
            self._register(version)
        self.assertEqual(
            [item["version"] for item in list_models(self.registry)],
            sorted(["v10", "v2", "v1", "2026.01", "2026.1", "a"]))
        item = list_models(self.registry)[0]
        self.assertEqual(set(item),
                         {"version", "status", "model_sha256", "manifest_sha256",
                          "ezkl_version", "output_scale", "artifacts"})

    def test_list_requires_an_existing_valid_registry(self):
        with self.assertRaises(RegistryError):
            list_models(self.registry)

    # -- lifecycle ----------------------------------------------------------

    def test_enable_disabled_and_revoked_transitions(self):
        self._register("v1")
        self.assertEqual(enable_model(self.registry, "v1"),
                         {"version": "v1", "status": "enabled", "changed": True})
        self.assertEqual(enable_model(self.registry, "v1"),
                         {"version": "v1", "status": "enabled", "changed": False})
        self.assertEqual(revoke_model(self.registry, "v1"),
                         {"version": "v1", "status": "revoked", "changed": True})
        self.assertEqual(revoke_model(self.registry, "v1"),
                         {"version": "v1", "status": "revoked", "changed": False})
        # Revocation is irreversible.
        with self.assertRaises(RegistryError):
            enable_model(self.registry, "v1")
        with self.assertRaises(RegistryError):
            revoke_model(self.registry, "missing")
        with self.assertRaises(RegistryError):
            enable_model(self.registry, "missing")

    def test_revoke_from_disabled_is_allowed(self):
        self._register("v1")
        self.assertEqual(revoke_model(self.registry, "v1")["status"], "revoked")

    def test_admit_only_returns_enabled_records(self):
        self._register("on", )
        self._register("off")
        enable_model(self.registry, "on")
        revoke_model(self.registry, "off")
        self.assertEqual(admit_version(self.registry, "on")["status"], "enabled")
        with self.assertRaises(RegistryError):
            admit_version(self.registry, "off")
        with self.assertRaises(RegistryError):
            admit_version(self.registry, "unknown")

    # -- corruption: reject and preserve -----------------------------------

    def test_corrupt_or_illegal_registry_is_rejected_and_preserved(self):
        self._register("v1")
        enable_model(self.registry, "v1")
        good_bytes = self.registry.read_bytes()
        record = load_registry(self.registry)["models"]["v1"]

        cases = {
            "broken": b"{not json",
            "notobject": json.dumps([1, 2]).encode(),
            "badversion": json.dumps({"format_version": 2, "kind": "model-registry",
                                      "models": {}}).encode(),
            "badkind": json.dumps({"format_version": 1, "kind": "x", "models": {}}).encode(),
            "extra": json.dumps({"format_version": 1, "kind": "model-registry",
                                 "models": {}, "x": 1}).encode(),
            "dupkey": b'{"format_version": 1, "kind": "model-registry",'
                      b' "models": {}, "models": {}}',
            "badstatus": json.dumps({"format_version": 1, "kind": "model-registry",
                                     "models": {"v1": {**record, "status": "x"}}}).encode(),
            "badfield": json.dumps({"format_version": 1, "kind": "model-registry",
                                    "models": {"v1": {**record, "output_scale": -1}}}).encode(),
            "badversionkey": json.dumps({"format_version": 1, "kind": "model-registry",
                                         "models": {"v 1": record}}).encode(),
        }
        for name, payload in cases.items():
            self.registry.write_bytes(payload)
            with self.assertRaises(RegistryError, msg=name):
                register_model(self.registry, "v9", self.manifest_path, self.model)
            with self.assertRaises(RegistryError, msg=name):
                list_models(self.registry)
            with self.assertRaises(RegistryError, msg=name):
                enable_model(self.registry, "v1")
            with self.assertRaises(RegistryError, msg=name):
                revoke_model(self.registry, "v1")
            # The corrupt file is byte-for-byte preserved.
            self.assertEqual(self.registry.read_bytes(), payload, msg=name)

        # A good registry is untouched by this test's failure cases.
        self.registry.write_bytes(good_bytes)
        self.assertEqual(admit_version(self.registry, "v1")["status"], "enabled")

    def test_no_temp_files_are_left_behind(self):
        for i in range(3):
            self._register(f"v{i}")
            enable_model(self.registry, f"v{i}")
        revoke_model(self.registry, "v1")
        leftovers = list(self.workspace.glob(".registry-*.tmp"))
        self.assertEqual(leftovers, [])

    # -- record/material comparison ----------------------------------------

    def test_require_record_matches_compares_every_pinned_field(self):
        self._register("v1")
        document = load_registry(self.registry)
        record = document["models"]["v1"]
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

        require_record_matches(record, manifest_sha=manifest_sha, manifest=manifest)

        for mutate in (
                lambda r: r.update(model_sha256=_digest(0xF1)),
                lambda r: r.update(manifest_sha256=_digest(0xF2)),
                lambda r: r.update(ezkl_version="9.9.9"),
                lambda r: r.update(output_scale=14),
                lambda r: r["artifacts"].update(settings=_digest(0xF3)),
                lambda r: r["artifacts"].update(vk=_digest(0xF4)),
                lambda r: r["artifacts"].update(srs=_digest(0xF5))):
            bad = copy.deepcopy(record)
            mutate(bad)
            with self.assertRaises(RegistryError):
                require_record_matches(bad, manifest_sha=manifest_sha, manifest=manifest)

    # -- CLI ----------------------------------------------------------------

    def _cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "zkml_quality", "model-registry", *args],
            cwd=str(ROOT), capture_output=True, text=True)

    def test_cli_lifecycle_and_outputs(self):
        proc = self._cli("register", "--registry", str(self.registry), "--version", "v1",
                         "--manifest", str(self.manifest_path), "--model", str(self.model))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(json.loads(proc.stdout)["status"], "disabled")

        # Idempotent register stays disabled.
        proc = self._cli("register", "--registry", str(self.registry), "--version", "v1",
                         "--manifest", str(self.manifest_path), "--model", str(self.model))
        self.assertEqual(json.loads(proc.stdout)["changed"], False)

        # Conflict.
        changed = self._write_manifest("m-cli.json", make_manifest(vk_nonce=9))
        proc = self._cli("register", "--registry", str(self.registry), "--version", "v1",
                         "--manifest", str(changed), "--model", str(self.model))
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("error:", proc.stderr)

        # Bad version token.
        proc = self._cli("enable", "--registry", str(self.registry), "--version", "bad v")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")

        proc = self._cli("enable", "--registry", str(self.registry), "--version", "v1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["changed"], True)

        self._cli("register", "--registry", str(self.registry), "--version", "v0",
                  "--manifest", str(self.manifest_path), "--model", str(self.model))
        proc = self._cli("list", "--registry", str(self.registry))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([item["version"] for item in json.loads(proc.stdout)["models"]],
                         ["v0", "v1"])

        proc = self._cli("revoke", "--registry", str(self.registry), "--version", "v1")
        self.assertEqual(json.loads(proc.stdout)["status"], "revoked")
        proc = self._cli("revoke", "--registry", str(self.registry), "--version", "v1")
        self.assertEqual(json.loads(proc.stdout)["changed"], False)

        # list on a missing registry fails nonzero with stderr only.
        proc = self._cli("list", "--registry", str(self.workspace / "nope.json"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
