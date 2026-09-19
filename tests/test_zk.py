"""End-to-end tests for the real EZKL 23.0.5 CPU proof loop."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from zkml_quality.inference import ROOT, infer
from zkml_quality.registry import enable_model, register_model, revoke_model
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
        cls.setup_info = run_setup(MODEL, cls.setup_dir)
        cls.model_sha = cls.setup_info["model_sha256"]
        cls.manifest = json.loads((cls.setup_dir / "manifest.json").read_text(encoding="utf-8"))

        # The verifier gets only public artifacts: manifest, model, settings,
        # VK and SRS. No proving key, no compiled circuit, never the private
        # input. The manifest is the verifier's own, independently obtained
        # root of trust.
        cls.verifier_dir = cls.workspace / "verifier"
        cls.verifier_dir.mkdir()
        shutil.copy(cls.setup_dir / "manifest.json", cls.verifier_dir / "manifest.json")
        shutil.copy(cls.setup_dir / "settings.json", cls.verifier_dir / "settings.json")
        shutil.copy(cls.setup_dir / "verification.key", cls.verifier_dir / "vk")
        shutil.copy(cls.setup_dir / "srs", cls.verifier_dir / "srs")
        shutil.copy(MODEL, cls.verifier_dir / "model.onnx")

        # The verifier's local registry records the trusted manifest and model
        # under a version they then enable; verification is gated on it.
        cls.registry_path = cls.workspace / "registry.json"
        cls.model_version = "1.0.0"
        register_model(cls.registry_path, cls.model_version,
                       cls.verifier_dir / "manifest.json", cls.verifier_dir / "model.onnx")
        enable_model(cls.registry_path, cls.model_version)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def _prove(self, name, values):
        input_path = self.workspace / f"{name}.json"
        input_path.write_text(json.dumps(features_document(values)), encoding="utf-8")
        credential_path = self.workspace / f"{name}.cred.json"
        run_prove(input_path, MODEL, self.setup_dir, credential_path)
        return json.loads(credential_path.read_text(encoding="utf-8")), credential_path

    def _verify(self, credential_path, manifest_path=None, **overrides):
        materials = {
            "model": self.verifier_dir / "model.onnx",
            "settings": self.verifier_dir / "settings.json",
            "vk": self.verifier_dir / "vk",
            "srs": self.verifier_dir / "srs",
            "registry": self.registry_path,
            "model_version": self.model_version,
        }
        materials.update(overrides)
        return run_verify(
            credential_path,
            self.verifier_dir / "manifest.json" if manifest_path is None else manifest_path,
            materials["model"], materials["settings"], materials["vk"], materials["srs"],
            materials["registry"], materials["model_version"])

    def _write_different_model(self, name="other.onnx"):
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        import numpy as np
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
        target = self.workspace / name
        onnx.save(model, target)
        return target

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
                self.assertEqual(
                    set(result),
                    {"verified", "model_sha256", "quantized_scores",
                     "scores_fixed_point", "label"})
                self.assertTrue(result["verified"])
                self.assertEqual(result["quantized_scores"], expected_quantized)
                self.assertEqual(result["label"], expected_label)
                self.assertEqual(result["model_sha256"], self.model_sha)
                self.assertNotIn("features", json.dumps(result))
                self.assertNotIn(str(self.workspace), json.dumps(result))

    def test_quantized_scores_are_the_circuit_outputs(self):
        # Proven values must not be confused with ordinary ONNX float scores:
        # they are integer field instances whose ordering matches `infer`.
        document = features_document([0.75, 0.875, 0.75, 0.625, 0.75, 0.875])
        float_result = infer(document)
        _, credential_path = self._prove("quantcheck", document["features"])
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

    def test_verify_requires_an_enabled_registered_version(self):
        _, credential_path = self._prove("gating", [0.125] * 6)

        # Unregistered version, invalid version string, missing registry.
        with self.assertRaises(ZkError):
            self._verify(credential_path, model_version="9.9.9")
        with self.assertRaises(ZkError):
            self._verify(credential_path, model_version="bad version!")
        with self.assertRaises(ZkError):
            self._verify(credential_path, registry=self.workspace / "no-registry.json")

        # Corrupt registry: rejected and left byte-for-byte untouched.
        corrupt = self.workspace / "corrupt-registry.json"
        corrupt.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ZkError):
            self._verify(credential_path, registry=corrupt)
        self.assertEqual(corrupt.read_text(encoding="utf-8"), "{not json")

        # Registered but still disabled: not admitted.
        gated_registry = self.workspace / "gated-registry.json"
        register_model(gated_registry, "2.0.0",
                       self.verifier_dir / "manifest.json", self.verifier_dir / "model.onnx")
        with self.assertRaises(ZkError):
            self._verify(credential_path, registry=gated_registry, model_version="2.0.0")

        # Revoked is terminal: verification stays refused and re-enabling fails.
        revoke_model(gated_registry, "2.0.0")
        with self.assertRaises(ZkError):
            self._verify(credential_path, registry=gated_registry, model_version="2.0.0")
        with self.assertRaises(ZkError):
            enable_model(gated_registry, "2.0.0")

        # A manifest/model pair the record does not pin is refused at the gate.
        evil = self._evil_copy("evil-gating")
        (evil / "model.onnx").write_bytes((evil / "model.onnx").read_bytes() + b"\x00")
        with self.assertRaises(ZkError):
            self._verify(credential_path, manifest_path=evil / "manifest.json",
                        model=evil / "model.onnx", settings=evil / "settings.json",
                        vk=evil / "vk", srs=evil / "srs")

        # The enabled record still verifies.
        self.assertTrue(self._verify(credential_path)["verified"])

    def test_manifest_pins_every_public_material(self):
        # The honest setup manifest records exactly the five artifact digests
        # plus the model digest, version and scale the verifier relies on.
        self.assertEqual(self.manifest["kind"], "zk-quality-setup")
        self.assertEqual(self.manifest["format_version"], 1)
        self.assertEqual(self.manifest["model_sha256"], self.model_sha)
        self.assertEqual(set(self.manifest["artifacts"]),
                         {"compiled", "settings", "pk", "vk", "srs"})

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

        # Rewriting the credential model digest alone cannot rebind the model:
        # the verifier compares against its own manifest, not the credential.
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

    def test_swapped_model_is_rejected_even_when_credential_agrees(self):
        other = self._write_different_model()
        _, credential_path = self._prove("wrongmodel", [0.125] * 6)

        # Different ONNX against the honest manifest: digest mismatch, and the
        # credential (which still names the original model) cannot override it.
        with self.assertRaises(ZkError):
            self._verify(credential_path, model=other)

        # The original attack: swap the model AND rewrite the credential's model
        # digest to match it. settings/VK/SRS stay the originals, so the honest
        # manifest still pins the original model and the credential claim is
        # rejected against it.
        other_sha = hashlib.sha256(other.read_bytes()).hexdigest()
        credential = json.loads(credential_path.read_text(encoding="utf-8"))
        forged = copy.deepcopy(credential)
        forged["model_sha256"] = other_sha
        forged_path = self.workspace / "forged-model-cred.json"
        forged_path.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaises(ZkError):
            self._verify(forged_path, model=other)

    def test_missing_or_malformed_manifest_is_rejected(self):
        _, credential_path = self._prove("badmanifest", [0.125] * 6)

        with self.assertRaises(ZkError):
            self._verify(credential_path,
                        manifest_path=self.workspace / "no-manifest.json")

        evil_dir = self.workspace / "evil-manifests"
        evil_dir.mkdir(exist_ok=True)

        def reject_manifest(doc, name):
            if isinstance(doc, (dict, list)):
                payload = json.dumps(doc)
            else:
                payload = doc
            path = evil_dir / name
            path.write_text(payload, encoding="utf-8")
            with self.assertRaises(ZkError):
                self._verify(credential_path, manifest_path=path)

        reject_manifest("{not json", "broken.json")
        good = copy.deepcopy(self.manifest)
        wrong_kind = copy.deepcopy(good)
        wrong_kind["kind"] = "zk-quality-credential"
        reject_manifest(wrong_kind, "kind.json")
        wrong_version = copy.deepcopy(good)
        wrong_version["format_version"] = 2
        reject_manifest(wrong_version, "format.json")
        wrong_ezkl = copy.deepcopy(good)
        wrong_ezkl["ezkl_version"] = "9.9.9"
        reject_manifest(wrong_ezkl, "ezkl.json")
        bad_model = copy.deepcopy(good)
        bad_model["model_sha256"] = "z" * 64
        reject_manifest(bad_model, "modelhex.json")
        bad_model2 = copy.deepcopy(good)
        bad_model2["model_sha256"] = "deadbeef"
        reject_manifest(bad_model2, "modelshort.json")
        missing_artifacts = copy.deepcopy(good)
        del missing_artifacts["artifacts"]["vk"]
        reject_manifest(missing_artifacts, "missingvk.json")
        bad_digest = copy.deepcopy(good)
        bad_digest["artifacts"]["srs"] = "0" * 63
        reject_manifest(bad_digest, "badsrs.json")
        extra_artifact = copy.deepcopy(good)
        extra_artifact["artifacts"]["extra"] = "a" * 64
        reject_manifest(extra_artifact, "extra.json")
        bad_scale = copy.deepcopy(good)
        bad_scale["output_scale"] = 0
        reject_manifest(bad_scale, "scale.json")
        reject_manifest(["not", "an", "object"], "notobject.json")

        # The untouched manifest still verifies.
        self.assertTrue(self._verify(credential_path)["verified"])

    def _evil_copy(self, name):
        """Copy the verifier materials into a scratch dir for tampering."""
        evil = self.workspace / name
        evil.mkdir(exist_ok=True)
        for filename in ("manifest.json", "settings.json", "vk", "srs", "model.onnx"):
            shutil.copy(self.verifier_dir / filename, evil / filename)
        return evil

    def test_tampered_manifest_digests_are_rejected(self):
        _, credential_path = self._prove("tampermanifest", [0.125] * 6)

        # Rewriting the pinned model digest in the manifest no longer matches
        # the real model file.
        evil = self._evil_copy("evil-model-digest")
        doc = json.loads((evil / "manifest.json").read_text(encoding="utf-8"))
        doc["model_sha256"] = "0" * 64
        (evil / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(ZkError):
            self._verify(credential_path, manifest_path=evil / "manifest.json",
                        model=evil / "model.onnx", settings=evil / "settings.json",
                        vk=evil / "vk", srs=evil / "srs")

        # A semantically altered settings file whose digest is echoed into the
        # manifest cannot be legitimized: the pinned output scale no longer
        # matches what the swapped settings encode (and the proof cannot verify
        # against the unchanged VK anyway).
        evil = self._evil_copy("evil-settings")
        swapped = evil / "settings.json"
        settings_doc = json.loads(swapped.read_text(encoding="utf-8"))
        settings_doc["model_output_scales"][0] += 1
        swapped.write_text(json.dumps(settings_doc), encoding="utf-8")
        doc = json.loads((evil / "manifest.json").read_text(encoding="utf-8"))
        doc["artifacts"]["settings"] = hashlib.sha256(swapped.read_bytes()).hexdigest()
        (evil / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(ZkError):
            self._verify(credential_path, manifest_path=evil / "manifest.json",
                        model=evil / "model.onnx", settings=evil / "settings.json",
                        vk=evil / "vk", srs=evil / "srs")

    def test_tampered_verification_files_are_rejected(self):
        _, credential_path = self._prove("tamperfiles", [0.125] * 6)

        for filename, override in (
                ("vk", "vk"), ("srs", "srs"), ("settings.json", "settings")):
            evil = self._evil_copy(f"evil-{override}")
            target = evil / filename
            target.write_bytes(target.read_bytes() + b"\x00")
            # Manifest left untouched: the recomputed digest cannot match.
            with self.assertRaises(ZkError):
                self._verify(credential_path, manifest_path=evil / "manifest.json",
                            model=evil / "model.onnx", settings=evil / "settings.json",
                            vk=evil / "vk", srs=evil / "srs")
            self.assertEqual(
                hashlib.sha256(target.read_bytes()).hexdigest()
                != self.manifest["artifacts"][override], True)

    def test_materials_from_a_different_setup_are_rejected(self):
        # A complete second setup for a different model: mixing any of its
        # VK/SRS/settings/manifest with the original proof must be rejected.
        other = self._write_different_model("other-setup.onnx")
        other_setup = self.workspace / "setup-other"
        run_setup(other, other_setup)
        _, credential_path = self._prove("mixed", [0.125] * 6)

        # Original manifest, foreign VK -> digest mismatch.
        with self.assertRaises(ZkError):
            self._verify(credential_path, vk=other_setup / "verification.key")
        # Original manifest, foreign SRS -> digest mismatch.
        with self.assertRaises(ZkError):
            self._verify(credential_path, srs=other_setup / "srs")
        # Foreign manifest + the whole foreign material set against the
        # original credential: the credential's model/artifact digests do not
        # match, and even forging them cannot make the original proof verify
        # against the foreign VK.
        with self.assertRaises(ZkError):
            self._verify(
                credential_path, manifest_path=other_setup / "manifest.json",
                model=other, settings=other_setup / "settings.json",
                vk=other_setup / "verification.key", srs=other_setup / "srs")

        # Forge every credential field to agree with the foreign manifest, but
        # keep the proof that was produced under the original circuit. All
        # manifest cross-checks now pass; the foreign VK must still reject it.
        other_manifest = json.loads(
            (other_setup / "manifest.json").read_text(encoding="utf-8"))
        original = json.loads(credential_path.read_text(encoding="utf-8"))
        forged = copy.deepcopy(original)
        forged["model_sha256"] = other_manifest["model_sha256"]
        forged["verification_artifacts"] = {
            "settings_sha256": other_manifest["artifacts"]["settings"],
            "vk_sha256": other_manifest["artifacts"]["vk"],
            "srs_sha256": other_manifest["artifacts"]["srs"],
        }
        forged["public_output"]["output_scale"] = other_manifest["output_scale"]
        forged["public_output"]["scores_fixed_point"] = [
            score / (1 << other_manifest["output_scale"])
            for score in forged["public_output"]["quantized_scores"]]
        forged_path = self.workspace / "forged-cross-credential.json"
        forged_path.write_text(json.dumps(forged), encoding="utf-8")
        with self.assertRaises(ZkError):
            self._verify(
                forged_path, manifest_path=other_setup / "manifest.json",
                model=other, settings=other_setup / "settings.json",
                vk=other_setup / "verification.key", srs=other_setup / "srs")

    def test_cli_failure_prints_only_stderr_and_nonzero(self):
        _, credential_path = self._prove("clitamper", [0.125] * 6)
        credential = json.loads(credential_path.read_text(encoding="utf-8"))
        credential["proof"]["proof"][40] ^= 0xFF
        bad = self.workspace / "cli-bad.json"
        bad.write_text(json.dumps(credential), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify",
             "--credential", str(bad),
             "--manifest", str(self.verifier_dir / "manifest.json"),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs"),
             "--registry", str(self.registry_path),
             "--model-version", self.model_version],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("error:", proc.stderr)

        # --manifest is mandatory: omitting it fails without running verify.
        proc = subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify",
             "--credential", str(bad),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs"),
             "--registry", str(self.registry_path),
             "--model-version", self.model_version],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("--manifest", proc.stderr)

        # --registry and --model-version are mandatory too.
        proc = subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify",
             "--credential", str(bad),
             "--manifest", str(self.verifier_dir / "manifest.json"),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs")],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("--registry", proc.stderr)

    def test_cli_success_prints_single_json(self):
        _, credential_path = self._prove("cligood", [0.125] * 6)
        proc = subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify",
             "--credential", str(credential_path),
             "--manifest", str(self.verifier_dir / "manifest.json"),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs"),
             "--registry", str(self.registry_path),
             "--model-version", self.model_version],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertTrue(result["verified"])
        self.assertEqual(result["label"], "normal")
        self.assertEqual(result["model_sha256"], self.model_sha)
        self.assertEqual(proc.stderr, "")

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
                       self.verifier_dir / "vk", self.verifier_dir / "srs",
                       self.registry_path, self.model_version)
        for missing in ("model.onnx", "settings.json", "vk", "srs"):
            with self.assertRaises(ZkError):
                run_verify(
                    self.workspace / "whatever.json",
                    self.verifier_dir / "manifest.json",
                    self.workspace / "missing-model" if missing == "model.onnx"
                    else self.verifier_dir / "model.onnx",
                    self.workspace / "missing-settings" if missing == "settings.json"
                    else self.verifier_dir / "settings.json",
                    self.workspace / "missing-vk" if missing == "vk"
                    else self.verifier_dir / "vk",
                    self.workspace / "missing-srs" if missing == "srs"
                    else self.verifier_dir / "srs",
                    self.registry_path, self.model_version)

    def test_setup_refuses_nonempty_directory(self):
        occupied = self.workspace / "occupied"
        occupied.mkdir()
        (occupied / "leftover").write_text("stale", encoding="utf-8")
        with self.assertRaises(ZkError):
            run_setup(MODEL, occupied)

    def test_setup_writes_complete_manifest_atomically(self):
        # A fresh setup directory contains the manifest alongside every
        # artifact it names, and the on-disk digests match.
        fresh = self.workspace / "setup-fresh"
        run_setup(MODEL, fresh)
        manifest = json.loads((fresh / "manifest.json").read_text(encoding="utf-8"))
        for key, filename in (
                ("compiled", "compiled.ezkl"),
                ("settings", "settings.json"),
                ("pk", "proving.key"),
                ("vk", "verification.key"),
                ("srs", "srs")):
            digest = hashlib.sha256((fresh / filename).read_bytes()).hexdigest()
            self.assertEqual(digest, manifest["artifacts"][key])
        self.assertFalse(list(fresh.glob(".*.part")))


if __name__ == "__main__":
    unittest.main()
