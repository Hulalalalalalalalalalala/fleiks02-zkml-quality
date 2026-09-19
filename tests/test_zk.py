"""End-to-end tests for the real EZKL 23.0.5 CPU proof loop."""
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from zkml_quality.inference import ROOT, infer
from zkml_quality.zk import (
    LABELS,
    ZkError,
    run_prove,
    run_setup,
    run_verify,
)

MODEL = ROOT / "models" / "quality.onnx"
SCALE = 13


def features_document(values):
    return {"features": list(values)}


class ZkLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(tempfile.mkdtemp(prefix="zktest-"))
        cls.setup_dir = cls.workspace / "setup"
        cls.model_sha = run_setup(MODEL, cls.setup_dir)["model_sha256"]

        # The verifier gets only public artifacts: no proving key, no compiled
        # circuit, and never the private input. The manifest is the verifier's
        # independently obtained trust root.
        cls.verifier_dir = cls.workspace / "verifier"
        cls.verifier_dir.mkdir()
        shutil.copy(cls.setup_dir / "settings.json", cls.verifier_dir / "settings.json")
        shutil.copy(cls.setup_dir / "verification.key", cls.verifier_dir / "vk")
        shutil.copy(cls.setup_dir / "srs", cls.verifier_dir / "srs")
        shutil.copy(cls.setup_dir / "manifest.json", cls.verifier_dir / "manifest.json")
        shutil.copy(MODEL, cls.verifier_dir / "model.onnx")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _prove(self, name, values):
        input_path = self.workspace / f"{name}.json"
        input_path.write_text(json.dumps(features_document(values)), encoding="utf-8")
        credential_path = self.workspace / f"{name}.cred.json"
        run_prove(input_path, MODEL, self.setup_dir, credential_path)
        return json.loads(credential_path.read_text(encoding="utf-8")), credential_path

    def _verify(self, credential_path, manifest=None):
        return run_verify(
            credential_path, manifest or self.verifier_dir / "manifest.json",
            self.verifier_dir / "model.onnx",
            self.verifier_dir / "settings.json",
            self.verifier_dir / "vk", self.verifier_dir / "srs")

    def test_real_loop_for_both_samples_and_tie(self):
        cases = (
            ("normal", [0.125] * 6, [6144, 2048], "normal"),
            ("inspect", [0.75, 0.875, 0.75, 0.625, 0.75, 0.875], [-14848, 23040], "inspect"),
            ("tie", [0, 0, 0, 0, 0.5, 1.0], [4096, 4096], "normal"),
        )
        for name, values, expected_quantized, expected_label in cases:
            with self.subTest(name=name):
                credential, credential_path = self._prove(name, values)
                self.assertEqual(credential["format_version"], 1)
                self.assertEqual(credential["kind"], "zk-quality-credential")
                self.assertEqual(credential["ezkl_version"], "23.0.5")
                self.assertEqual(credential["model_sha256"], self.model_sha)
                public = credential["public_output"]
                self.assertEqual(public["quantized_scores"], expected_quantized)
                self.assertEqual(public["label"], expected_label)
                self.assertEqual(public["output_scale"], SCALE)
                self.assertEqual(public["scores_fixed_point"],
                                 [value / (1 << SCALE) for value in expected_quantized])
                self.assertGreater(len(credential["proof"]["proof"]), 100)

                result = self._verify(credential_path)
                self.assertTrue(result["verified"])
                self.assertEqual(result["quantized_scores"], expected_quantized)
                self.assertEqual(result["label"], expected_label)
                self.assertEqual(result["model_sha256"], self.model_sha)

    def test_quantized_scores_are_the_circuit_outputs(self):
        # Proven values must not be confused with ordinary ONNX float scores:
        # they are integer field instances whose ordering matches `infer`.
        document = features_document([0.75, 0.875, 0.75, 0.625, 0.75, 0.875])
        float_result = infer(document)
        credential, credential_path = self._prove("quantcheck", document["features"])
        proven = self._verify(credential_path)
        self.assertEqual(proven["label"], float_result["label"])
        for proven_value, float_score in zip(proven["scores_fixed_point"], float_result["scores"]):
            self.assertAlmostEqual(proven_value, float_score, places=5)

    def test_credential_never_carries_private_input_or_paths(self):
        credential, _ = self._prove("privacy", [0.125] * 6)
        raw = json.dumps(credential)
        self.assertNotIn("features", raw)
        self.assertNotIn("input_data", raw)
        self.assertNotIn(str(self.workspace), raw)
        self.assertNotIn("samples", raw)

    def test_verifier_does_not_need_proving_key_or_compiled_circuit(self):
        _, credential_path = self._prove("verifieronly", [0.125] * 6)
        self.assertFalse((self.verifier_dir / "proving.key").exists())
        self.assertFalse((self.verifier_dir / "compiled.ezkl").exists())
        self.assertTrue(self._verify(credential_path)["verified"])

    def test_tampered_proof_scores_label_and_digests_are_rejected(self):
        credential, credential_path = self._prove("tamper", [0.125] * 6)

        def reject(mutated, message):
            bad_path = self.workspace / f"bad-{message}.json"
            bad_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaises(ZkError):
                self._verify(bad_path)

        mutated = copy.deepcopy(credential)
        mutated["proof"]["proof"][40] ^= 0xFF
        reject(mutated, "proof-byte")

        mutated = copy.deepcopy(credential)
        mutated["public_output"]["quantized_scores"][0] += 1
        reject(mutated, "score")

        mutated = copy.deepcopy(credential)
        mutated["public_output"]["label"] = next(
            label for label in LABELS if label != credential["public_output"]["label"])
        reject(mutated, "label")

        mutated = copy.deepcopy(credential)
        mutated["model_sha256"] = "0" * 64
        reject(mutated, "model-hash")

        mutated = copy.deepcopy(credential)
        mutated["verification_artifacts"]["vk_sha256"] = "f" * 64
        reject(mutated, "vk-digest")

        mutated = copy.deepcopy(credential)
        mutated["proof"]["instances"][0][0] = "0019000000000000000000000000000000000000000000000000000000000000"
        reject(mutated, "instance")

        # The original credential still verifies.
        self.assertTrue(self._verify(credential_path)["verified"])

    def test_wrong_model_is_rejected(self):
        self._write_different_model()
        _, credential_path = self._prove("wrongmodel", [0.125] * 6)
        with self.assertRaises(ZkError):
            run_verify(credential_path, self.verifier_dir / "manifest.json",
                       self.workspace / "other.onnx",
                       self.verifier_dir / "settings.json",
                       self.verifier_dir / "vk", self.verifier_dir / "srs")

    def test_manifest_is_the_trust_root(self):
        credential, credential_path = self._prove("trustroot", [0.125] * 6)
        manifest = json.loads((self.verifier_dir / "manifest.json").read_text(encoding="utf-8"))

        def reject_manifest(mutated, message):
            bad_path = self.workspace / f"bad-manifest-{message}.json"
            bad_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaises(ZkError, msg=message):
                self._verify(credential_path, manifest=bad_path)

        # Missing, malformed and unsupported manifests are rejected.
        with self.assertRaises(ZkError):
            self._verify(credential_path, manifest=self.workspace / "no-manifest.json")
        reject_manifest({**manifest, "kind": "other"}, "kind")
        reject_manifest({**manifest, "format_version": 999}, "version")
        reject_manifest({**manifest, "ezkl_version": "0.0.0"}, "ezkl")
        reject_manifest({**manifest, "model_sha256": "not-a-digest"}, "model-digest")
        reject_manifest({**manifest, "artifacts": {"settings": "z" * 64}}, "artifact-digest")

        # A manifest that does not match the physical material is rejected.
        mutated = copy.deepcopy(manifest)
        mutated["model_sha256"] = "0" * 64
        reject_manifest(mutated, "model-mismatch")
        for key in ("settings", "vk", "srs"):
            mutated = copy.deepcopy(manifest)
            mutated["artifacts"][key] = "1" * 64
            reject_manifest(mutated, f"{key}-mismatch")

        # Rewriting the credential's claims to match a swapped-in model and
        # re-anchored digests cannot relocate the trust root: the verifier's
        # manifest still pins the original model and artifacts.
        self._write_different_model()
        other_sha = hashlib.sha256(
            (self.workspace / "other.onnx").read_bytes()).hexdigest()
        mutated = copy.deepcopy(credential)
        mutated["model_sha256"] = other_sha
        bad_path = self.workspace / "bad-rewritten-claims.json"
        bad_path.write_text(json.dumps(mutated), encoding="utf-8")
        with self.assertRaises(ZkError):
            run_verify(bad_path, self.verifier_dir / "manifest.json",
                       self.workspace / "other.onnx",
                       self.verifier_dir / "settings.json",
                       self.verifier_dir / "vk", self.verifier_dir / "srs")

        # The genuine manifest and credential still verify.
        self.assertTrue(self._verify(credential_path)["verified"])

    def _write_different_model(self):
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        w1 = np.eye(6, 4, dtype=np.float32)
        w2 = np.zeros((4, 2), dtype=np.float32)
        initializers = [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(np.zeros(4, dtype=np.float32), "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(np.zeros(2, dtype=np.float32), "b2"),
        ]
        nodes = [
            helper.make_node("MatMul", ["features", "w1"], ["h0"]),
            helper.make_node("Add", ["h0", "b1"], ["h1"]),
            helper.make_node("Relu", ["h1"], ["h2"]),
            helper.make_node("MatMul", ["h2", "w2"], ["o0"]),
            helper.make_node("Add", ["o0", "b2"], ["scores"]),
        ]
        graph = helper.make_graph(
            nodes, "other",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 6])],
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])],
            initializers)
        model = helper.make_model(
            graph, producer_name="other",
            opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.save(model, self.workspace / "other.onnx")

    def test_invalid_inputs_and_paths(self):
        bad_inputs = (
            [0] * 5, [0] * 7, [True] * 6,
            [float("nan")] * 6, [1.1] * 6, [-0.1] * 6,
        )
        for values in bad_inputs:
            input_path = self.workspace / "bad-input.json"
            input_path.write_text(json.dumps(features_document(values)), encoding="utf-8")
            with self.assertRaises(ZkError):
                run_prove(input_path, MODEL, self.setup_dir,
                          self.workspace / "never-written.json")
        with self.assertRaises(ZkError):
            run_prove(self.workspace / "missing-input.json", MODEL, self.setup_dir,
                      self.workspace / "never-written.json")
        with self.assertRaises(ZkError):
            run_prove(MODEL, MODEL, self.setup_dir, self.workspace / "never-written.json")
        with self.assertRaises(ZkError):
            run_verify(self.workspace / "missing-credential.json",
                       self.verifier_dir / "manifest.json",
                       MODEL, self.verifier_dir / "settings.json",
                       self.verifier_dir / "vk", self.verifier_dir / "srs")

    def test_setup_refuses_nonempty_directory(self):
        occupied = self.workspace / "occupied"
        occupied.mkdir()
        (occupied / "leftover").write_text("stale", encoding="utf-8")
        with self.assertRaises(ZkError):
            run_setup(MODEL, occupied)


if __name__ == "__main__":
    unittest.main()
