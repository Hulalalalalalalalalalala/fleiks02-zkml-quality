"""End-to-end tests for the offline zk-verify-batch command.

The batch command performs the same real EZKL 23.0.5 verifications as
zk-verify: nothing is mocked. A small fixed set of credentials is proven once
for the whole class; per-test copies are tampered to exercise the three
per-item rejection codes.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from zkml_quality.inference import ROOT
from zkml_quality.registry import enable_model, register_model, revoke_model
from zkml_quality.verify_batch import (
    CODE_CREDENTIAL_INVALID,
    CODE_CREDENTIAL_MISSING,
    CODE_TRUST_FAILED,
    CODE_VERIFICATION_FAILED,
    BatchError,
    verify_batch,
)
from zkml_quality.zk import run_prove, run_setup

MODEL = ROOT / "models" / "quality.onnx"
VERSION = "v1.0"

ACCEPTED_KEYS = {
    "id", "accepted", "model_sha256",
    "quantized_scores", "scores_fixed_point", "label",
}
REJECTED_KEYS = {"id", "accepted", "error"}
ERROR_KEYS = {"code", "message"}


class VerifyBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(tempfile.mkdtemp(prefix="zkbatch-"))
        cls.setup_dir = cls.workspace / "setup"
        cls.setup_info = run_setup(MODEL, cls.setup_dir)
        cls.model_sha = cls.setup_info["model_sha256"]

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

        # Three real credentials, proven once and reused read-only.
        cls.normal_path = cls._prove_credential(
            cls, "normal", [0.125] * 6)
        cls.inspect_path = cls._prove_credential(
            cls, "inspect", [0.75, 0.875, 0.75, 0.625, 0.75, 0.875])
        cls.tie_path = cls._prove_credential(
            cls, "tie", [0, 0, 0, 0, 0.5, 1.0])

    def _prove_credential(self, name, values):
        input_path = self.workspace / f"{name}-input.json"
        input_path.write_text(json.dumps({"features": list(values)}), encoding="utf-8")
        credential_path = self.workspace / f"{name}.cred.json"
        run_prove(input_path, MODEL, self.setup_dir, credential_path)
        return credential_path

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    # -- helpers -----------------------------------------------------------

    def _write_credential(self, name, document):
        path = self.workspace / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _tampered(self, name, mutate):
        credential = json.loads(self.normal_path.read_text(encoding="utf-8"))
        mutate(credential)
        return self._write_credential(name, credential)

    def _descriptor(self, items, name="batch.json"):
        path = self.workspace / name
        path.write_text(json.dumps({"items": items}), encoding="utf-8")
        return path

    def _item(self, identifier, credential_path):
        return {"id": identifier, "credential": str(credential_path)}

    def _run_batch(self, descriptor, *, registry=None, version=VERSION,
                   manifest=None, model=None, settings=None, vk=None, srs=None):
        return verify_batch(
            descriptor,
            self.verifier_dir / "manifest.json" if manifest is None else manifest,
            self.verifier_dir / "model.onnx" if model is None else model,
            self.verifier_dir / "settings.json" if settings is None else settings,
            self.verifier_dir / "vk" if vk is None else vk,
            self.verifier_dir / "srs" if srs is None else srs,
            self.registry if registry is None else registry,
            version)

    def _cli(self, descriptor, *extra):
        return subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify-batch",
             "--file", str(descriptor),
             "--registry", str(self.registry), "--model-version", VERSION,
             "--manifest", str(self.verifier_dir / "manifest.json"),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs"), *extra],
            cwd=str(ROOT), capture_output=True, text=True)

    # -- happy paths --------------------------------------------------------

    def test_batch_accepts_all_real_credentials_in_order(self):
        descriptor = self._descriptor([
            self._item("a-normal", self.normal_path),
            self._item("b-inspect", self.inspect_path),
            self._item("c.tie-1", self.tie_path),
        ], name="all-good.json")
        result = self._run_batch(descriptor)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["accepted"], 3)
        self.assertEqual(result["rejected"], 0)
        self.assertEqual([entry["id"] for entry in result["results"]],
                         ["a-normal", "b-inspect", "c.tie-1"])
        for entry in result["results"]:
            self.assertEqual(set(entry), ACCEPTED_KEYS)
            self.assertIs(entry["accepted"], True)
            self.assertEqual(entry["model_sha256"], self.model_sha)
        self.assertEqual(result["results"][0]["quantized_scores"], [6144, 2048])
        self.assertEqual(result["results"][0]["label"], "normal")
        self.assertEqual(result["results"][1]["quantized_scores"], [-14848, 23040])
        self.assertEqual(result["results"][1]["label"], "inspect")
        # Tie keeps the same quantization/tie rule as infer and zk-verify.
        self.assertEqual(result["results"][2]["quantized_scores"], [4096, 4096])
        self.assertEqual(result["results"][2]["label"], "normal")

    def test_cli_success_is_one_stdout_json_exit_zero(self):
        descriptor = self._descriptor([self._item("only", self.normal_path)],
                                      name="cli-good.json")
        proc = self._cli(descriptor)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        result = json.loads(proc.stdout)
        self.assertEqual(set(result), {"total", "accepted", "rejected", "results"})
        self.assertEqual((result["total"], result["accepted"], result["rejected"]),
                         (1, 1, 0))

    # -- per-item rejection classification ----------------------------------

    def test_mixed_batch_rejects_only_bad_items_and_still_exits_zero(self):
        missing = self.workspace / "no-such-credential.json"
        malformed = self.workspace / "malformed.json"
        malformed.write_text("{not json", encoding="utf-8")
        not_object = self.workspace / "not-object.json"
        not_object.write_text(json.dumps(["nope"]), encoding="utf-8")
        bad_proof = self._tampered(
            "bad-proof.json", lambda c: c["proof"]["proof"].__setitem__(40, c["proof"]["proof"][40] ^ 0xFF))
        bad_instance = self._tampered(
            "bad-instance.json",
            lambda c: c["proof"]["instances"][0].__setitem__(
                0, "0019000000000000000000000000000000000000000000000000000000000000"))
        bad_score = self._tampered(
            "bad-score.json",
            lambda c: (
                c["public_output"]["quantized_scores"].__setitem__(
                    0, c["public_output"]["quantized_scores"][0] + 1),
                c["public_output"].__setitem__(
                    "scores_fixed_point",
                    [s / (1 << c["public_output"]["output_scale"])
                     for s in c["public_output"]["quantized_scores"]])))
        bad_label = self._tampered(
            "bad-label.json",
            lambda c: c["public_output"].__setitem__(
                "label", "inspect" if c["public_output"]["label"] == "normal" else "normal"))
        bad_model_digest = self._tampered(
            "bad-model-digest.json", lambda c: c.__setitem__("model_sha256", "0" * 64))
        bad_vk_digest = self._tampered(
            "bad-vk-digest.json",
            lambda c: c["verification_artifacts"].__setitem__("vk_sha256", "f" * 64))
        directory = self.workspace / "a-directory"
        directory.mkdir(exist_ok=True)

        descriptor = self._descriptor([
            self._item("good-1", self.normal_path),
            self._item("missing-1", missing),
            self._item("missing-dir", directory),
            self._item("malformed-1", malformed),
            self._item("malformed-2", not_object),
            self._item("bad-proof", bad_proof),
            self._item("bad-instance", bad_instance),
            self._item("bad-score", bad_score),
            self._item("bad-label", bad_label),
            self._item("bad-model-digest", bad_model_digest),
            self._item("bad-vk-digest", bad_vk_digest),
            self._item("good-2", self.inspect_path),
        ], name="mixed.json")

        result = self._run_batch(descriptor)
        self.assertEqual(result["total"], 12)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["rejected"], 10)
        by_id = {entry["id"]: entry for entry in result["results"]}

        def rejected(identifier, code):
            entry = by_id[identifier]
            self.assertEqual(set(entry), REJECTED_KEYS, identifier)
            self.assertIs(entry["accepted"], False)
            self.assertEqual(set(entry["error"]), ERROR_KEYS)
            self.assertEqual(entry["error"]["code"], code)
            # Fixed, non-revealing wording; never exception text or a path.
            self.assertIsInstance(entry["error"]["message"], str)
            self.assertTrue(entry["error"]["message"])
            self.assertNotIn(str(self.workspace), entry["error"]["message"])

        self.assertIs(by_id["good-1"]["accepted"], True)
        self.assertIs(by_id["good-2"]["accepted"], True)
        rejected("missing-1", CODE_CREDENTIAL_MISSING)
        rejected("missing-dir", CODE_CREDENTIAL_MISSING)
        rejected("malformed-1", CODE_CREDENTIAL_INVALID)
        rejected("malformed-2", CODE_CREDENTIAL_INVALID)
        rejected("bad-proof", CODE_VERIFICATION_FAILED)
        rejected("bad-instance", CODE_VERIFICATION_FAILED)
        rejected("bad-score", CODE_VERIFICATION_FAILED)
        rejected("bad-label", CODE_VERIFICATION_FAILED)
        rejected("bad-model-digest", CODE_CREDENTIAL_INVALID)
        rejected("bad-vk-digest", CODE_CREDENTIAL_INVALID)

    def test_all_items_rejected_still_exit_zero_and_ordered(self):
        descriptor = self._descriptor([
            self._item("x1", self.workspace / "missing-a.json"),
            self._item("x2", self.workspace / "missing-b.json"),
        ], name="all-bad.json")
        result = self._run_batch(descriptor)
        self.assertEqual((result["total"], result["accepted"], result["rejected"]),
                         (2, 0, 2))
        self.assertEqual([entry["id"] for entry in result["results"]], ["x1", "x2"])

    # -- descriptor validation: whole-batch rejection -----------------------

    def _assert_invalid_descriptor(self, payload, *, name, raw=False):
        path = self.workspace / name
        path.write_text(payload if raw else json.dumps(payload), encoding="utf-8")
        with self.assertRaises(BatchError) as caught:
            self._run_batch(path)
        self.assertEqual(caught.exception.code, "invalid_batch")

    def test_descriptor_must_be_object_with_only_nonempty_items(self):
        self._assert_invalid_descriptor([], name="d-array.json")
        self._assert_invalid_descriptor({}, name="d-empty.json")
        self._assert_invalid_descriptor({"items": []}, name="d-noitems.json")
        self._assert_invalid_descriptor({"items": {}}, name="d-objectitems.json")
        self._assert_invalid_descriptor(
            {"items": [self._item("ok", self.normal_path)], "extra": 1},
            name="d-extra.json")
        self._assert_invalid_descriptor(
            {"items": "nope"}, name="d-stritems.json")
        self._assert_invalid_descriptor("not json at all", name="d-broken.json", raw=True)

    def test_descriptor_items_must_have_only_id_and_credential(self):
        good = self._item("ok", self.normal_path)
        self._assert_invalid_descriptor(
            {"items": [["not", "an", "object"]]}, name="i-array.json")
        self._assert_invalid_descriptor({"items": ["nope"]}, name="i-string.json")
        missing_id = {"credential": good["credential"]}
        self._assert_invalid_descriptor({"items": [missing_id]}, name="i-noid.json")
        missing_cred = {"id": "ok"}
        self._assert_invalid_descriptor({"items": [missing_cred]}, name="i-nocred.json")
        extra = dict(good, extra=1)
        self._assert_invalid_descriptor({"items": [extra]}, name="i-extra.json")
        self._assert_invalid_descriptor(
            {"items": [{"id": "ok", "credential": ""}]}, name="i-emptycred.json")
        self._assert_invalid_descriptor(
            {"items": [{"id": 7, "credential": good["credential"]}]},
            name="i-intid.json")
        self._assert_invalid_descriptor(
            {"items": [{"id": None, "credential": good["credential"]}]},
            name="i-nullid.json")
        self._assert_invalid_descriptor(
            {"items": [{"id": "ok", "credential": 9}]}, name="i-intcred.json")

    def test_descriptor_ids_must_match_token_and_be_unique(self):
        good_cred = str(self.normal_path)
        for bad_id in ("", "bad id", "slash/x", "a:b", "émile", "a/b"):
            self._assert_invalid_descriptor(
                {"items": [{"id": bad_id, "credential": good_cred}]},
                name=f"id-{len(bad_id)}-{ord(bad_id[0]) if bad_id else 0}.json")
        self._assert_invalid_descriptor(
            {"items": [
                {"id": "dup", "credential": good_cred},
                {"id": "dup", "credential": good_cred}]},
            name="id-dup.json")

    def test_descriptor_duplicate_json_keys_are_rejected(self):
        good_cred = str(self.normal_path)
        # Duplicate top-level "items" key.
        payload = (
            '{"items": [{"id": "a", "credential": '
            + json.dumps(good_cred)
            + '}], "items": [{"id": "b", "credential": '
            + json.dumps(good_cred) + '}]}')
        self._assert_invalid_descriptor(payload, name="dup-items-key.json", raw=True)
        # Duplicate key inside an item.
        payload = (
            '{"items": [{"id": "a", "credential": '
            + json.dumps(good_cred)
            + ', "id": "a2"}]}')
        self._assert_invalid_descriptor(payload, name="dup-item-key.json", raw=True)

    def test_descriptor_size_bounds(self):
        # 257 items is rejected before the trust gate or any file is opened.
        oversized = {"items": [
            {"id": f"id-{index}", "credential": str(self.workspace / f"c-{index}.json")}
            for index in range(257)]}
        self._assert_invalid_descriptor(oversized, name="too-many.json")
        # 256 items passes descriptor validation; it then reaches the trust
        # gate (which fails here without touching credentials), proving the
        # boundary itself is accepted.
        at_limit = self._descriptor([
            {"id": f"id-{index:03d}",
             "credential": str(self.workspace / f"c-{index}.json")}
            for index in range(256)], name="at-limit.json")
        with self.assertRaises(BatchError) as caught:
            self._run_batch(at_limit, registry=self.workspace / "no-reg.json")
        self.assertEqual(caught.exception.code, CODE_TRUST_FAILED)

    def test_bad_descriptor_is_rejected_even_when_trust_also_broken(self):
        # Descriptor validation runs first: an invalid descriptor with a
        # missing registry must surface as invalid_batch, not a trust failure.
        descriptor = self.workspace / "ordering.json"
        descriptor.write_text("{not json", encoding="utf-8")
        with self.assertRaises(BatchError) as caught:
            self._run_batch(descriptor, registry=self.workspace / "no-reg.json")
        self.assertEqual(caught.exception.code, "invalid_batch")

    # -- one-time trust gate ------------------------------------------------

    def test_trust_gate_failure_terminates_the_whole_batch(self):
        good_items = [self._item("good", self.normal_path)]
        descriptor = self._descriptor(good_items, name="trust.json")

        # Unknown / disabled / revoked versions and a corrupt registry.
        with self.assertRaises(BatchError) as caught:
            self._run_batch(descriptor, version="not-registered")
        self.assertEqual(caught.exception.code, CODE_TRUST_FAILED)

        disabled_reg = self.workspace / "reg-disabled.json"
        register_model(disabled_reg, "d1",
                       self.verifier_dir / "manifest.json",
                       self.verifier_dir / "model.onnx")
        with self.assertRaises(BatchError):
            self._run_batch(descriptor, registry=disabled_reg, version="d1")

        revoked_reg = self.workspace / "reg-revoked.json"
        register_model(revoked_reg, "r1",
                       self.verifier_dir / "manifest.json",
                       self.verifier_dir / "model.onnx")
        enable_model(revoked_reg, "r1")
        revoke_model(revoked_reg, "r1")
        with self.assertRaises(BatchError):
            self._run_batch(descriptor, registry=revoked_reg, version="r1")

        corrupt_reg = self.workspace / "reg-corrupt.json"
        corrupt_reg.write_text("{not json", encoding="utf-8")
        with self.assertRaises(BatchError):
            self._run_batch(descriptor, registry=corrupt_reg)

        with self.assertRaises(BatchError):
            self._run_batch(descriptor, registry=self.workspace / "missing-reg.json")

        # A tampered trusted material (VK bytes no longer match the manifest).
        evil = self.workspace / "evil-vk"
        evil.write_bytes((self.verifier_dir / "vk").read_bytes() + b"\x00")
        with self.assertRaises(BatchError):
            self._run_batch(descriptor, vk=evil)

        # A missing trusted material.
        with self.assertRaises(BatchError):
            self._run_batch(descriptor, srs=self.workspace / "missing-srs")

    def test_trust_failure_runs_no_item_verifications(self):
        # Even items that would all reject as missing are never classified:
        # a trust failure aborts before the first credential is opened.
        descriptor = self._descriptor(
            [self._item("whatever", self.workspace / "gone.json")],
            name="early-abort.json")
        with self.assertRaises(BatchError):
            self._run_batch(descriptor, version="not-registered")

    # -- CLI envelope --------------------------------------------------------

    def test_cli_bad_descriptor_is_nonzero_empty_stdout_one_stderr_json(self):
        descriptor = self.workspace / "cli-bad-descriptor.json"
        descriptor.write_text('{"items": []}', encoding="utf-8")
        proc = self._cli(descriptor)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        envelope = json.loads(proc.stderr)
        self.assertEqual(set(envelope), {"error"})
        self.assertEqual(envelope["error"]["code"], "invalid_batch")
        self.assertTrue(envelope["error"]["message"])
        self.assertNotIn(str(self.workspace), proc.stderr)

    def test_cli_trust_failure_is_nonzero_empty_stdout_one_stderr_json(self):
        descriptor = self._descriptor(
            [self._item("good", self.normal_path)], name="cli-trust.json")
        proc = subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify-batch",
             "--file", str(descriptor),
             "--registry", str(self.workspace / "no-reg.json"),
             "--model-version", VERSION,
             "--manifest", str(self.verifier_dir / "manifest.json"),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs")],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        envelope = json.loads(proc.stderr)
        self.assertEqual(envelope["error"]["code"], CODE_TRUST_FAILED)

    def test_cli_mixed_batch_exits_zero_with_one_json_and_safe_content(self):
        bad_proof = self._tampered(
            "cli-bad-proof.json",
            lambda c: c["proof"]["proof"].__setitem__(40, c["proof"]["proof"][40] ^ 0xFF))
        descriptor = self._descriptor([
            self._item("good", self.inspect_path),
            self._item("missing", self.workspace / "nope.json"),
            self._item("bad-proof", bad_proof),
        ], name="cli-mixed.json")
        proc = self._cli(descriptor)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        result = json.loads(proc.stdout)
        self.assertEqual((result["total"], result["accepted"], result["rejected"]),
                         (3, 1, 2))
        # Output is one JSON object: a second parse of the same stream fails.
        decoder = json.JSONDecoder()
        _, end = decoder.raw_decode(proc.stdout)
        self.assertEqual(proc.stdout[end:].strip(), "")

    # -- privacy / side effects ----------------------------------------------

    def test_output_never_contains_paths_features_inputs_or_proofs(self):
        bad_proof = self._tampered(
            "privacy-bad-proof.json",
            lambda c: c["proof"]["proof"].__setitem__(40, c["proof"]["proof"][40] ^ 0xFF))
        descriptor = self._descriptor([
            self._item("ok", self.normal_path),
            self._item("tampered", bad_proof),
        ], name="privacy.json")
        result = self._run_batch(descriptor)
        rendered = json.dumps(result)
        # No paths, private inputs or proof/blob material ever leave the
        # command. The fixed English error wording is allowed to mention the
        # word "proof"; the proof bytes themselves are not present.
        original = json.loads(self.normal_path.read_text(encoding="utf-8"))
        proof_blob = json.dumps(original["proof"]["proof"])
        self.assertNotIn(str(self.workspace), rendered)
        self.assertNotIn(proof_blob, rendered)
        for entry in result["results"]:
            self.assertNotIn("features", entry)
            self.assertNotIn("input_data", entry)
            self.assertNotIn("proof", entry)
            self.assertNotIn("verification_artifacts", entry)
        # Rejected entries carry exactly id/accepted/error.code/error.message.
        rejected_entry = result["results"][1]
        self.assertEqual(set(rejected_entry), REJECTED_KEYS)
        self.assertEqual(set(rejected_entry["error"]), ERROR_KEYS)

    def test_batch_is_offline_and_writes_nothing(self):
        descriptor = self._descriptor([
            self._item("ok", self.normal_path),
            self._item("missing", self.workspace / "gone.json"),
        ], name="nowrite.json")

        def snapshot():
            return {str(p.relative_to(self.workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.workspace.rglob("*") if p.is_file()}

        before = snapshot()
        self._run_batch(descriptor)
        after = snapshot()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
