import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from zkml_quality.inference import ROOT
from zkml_quality.zk import (
    CREDENTIAL_FORMAT,
    REQUIRED_EZKL_VERSION,
    ZkError,
    run_prove,
    run_setup,
    run_verify,
)

MODEL = ROOT / "models" / "quality.onnx"


class ZkPipelineTest(unittest.TestCase):
    """The real EZKL loop is slow (a 138 MB proving key), so share one setup."""

    setup_dir = None
    temp_root = None

    @classmethod
    def setUpClass(cls):
        cls.temp_root = tempfile.TemporaryDirectory(prefix="zkml-tests-")
        root = Path(cls.temp_root.name)
        cls.setup_dir = root / "setup"
        run_setup(MODEL, cls.setup_dir)
        cls.credentials = {}
        for name, features in (
            ("normal", [0.125] * 6),
            ("inspect", [0.75, 0.875, 0.75, 0.625, 0.75, 0.875]),
            ("tie", [0.0, 0.0, 0.0, 0.0, 0.5, 1.0]),
        ):
            input_path = root / f"{name}.json"
            input_path.write_text(json.dumps({"features": features}))
            credential_path = root / f"{name}.cred.json"
            run_prove(input_path, MODEL, cls.setup_dir, credential_path)
            cls.credentials[name] = credential_path

    @classmethod
    def tearDownClass(cls):
        cls.temp_root.cleanup()

    def _artifacts(self):
        return {
            "settings_path": self.setup_dir / "settings.json",
            "vk_path": self.setup_dir / "vk.key",
            "srs_path": self.setup_dir / "kzg.srs",
        }

    def test_full_loop_and_proven_outputs(self):
        expected = {"normal": ([0.75, 0.25], "normal"),
                    "inspect": ([-1.8125, 2.8125], "inspect"),
                    "tie": ([0.5, 0.5], "normal")}
        for name, (scores, label) in expected.items():
            result = run_verify(self.credentials[name], MODEL, **self._artifacts())
            self.assertTrue(result["verified"])
            self.assertEqual(result["label"], label)
            self.assertEqual(result["quantized_scores"], scores)
            self.assertEqual(len(result["model_sha256"]), 64)

    def test_credential_schema_and_privacy(self):
        credential = json.loads(self.credentials["normal"].read_text())
        self.assertEqual(set(credential), {
            "format_version", "ezkl_version", "model_sha256", "artifacts",
            "proof", "public_output"})
        self.assertEqual(credential["format_version"], CREDENTIAL_FORMAT)
        self.assertEqual(credential["ezkl_version"], REQUIRED_EZKL_VERSION)
        self.assertEqual(credential["model_sha256"], self.setup_manifest_model_hash())
        # No raw features, input contents, or local paths anywhere in the file.
        blob = self.credentials["normal"].read_text()
        for banned in ("features", "0.125", str(self.temp_root.name), str(ROOT)):
            self.assertNotIn(banned, blob)
        proof = json.loads(base64.b64decode(credential["proof"]["data"]))
        # Only the two scores are public; six private inputs are never exposed.
        self.assertEqual(len(proof["instances"][0]), 2)

    def setup_manifest_model_hash(self):
        return json.loads((self.setup_dir / "manifest.json").read_text())["model_sha256"]

    def test_tampered_proof_fails(self):
        credential = json.loads(self.credentials["normal"].read_text())
        raw = bytearray(base64.b64decode(credential["proof"]["data"]))
        raw[300] ^= 0x01
        credential["proof"]["data"] = base64.b64encode(bytes(raw)).decode()
        with self._temp_credential(credential) as path:
            with self.assertRaises(ZkError):
                run_verify(path, MODEL, **self._artifacts())

    def test_forged_scores_and_label_fail(self):
        credential = json.loads(self.credentials["normal"].read_text())
        forged = copy.deepcopy(credential)
        forged["public_output"]["quantized_scores"] = [9.9, 0.1]
        with self._temp_credential(forged) as path:
            with self.assertRaises(ZkError):
                run_verify(path, MODEL, **self._artifacts())
        forged = copy.deepcopy(credential)
        forged["public_output"]["label"] = "inspect"
        with self._temp_credential(forged) as path:
            with self.assertRaises(ZkError):
                run_verify(path, MODEL, **self._artifacts())

    def test_wrong_or_tampered_artifacts_fail(self):
        good = self.credentials["normal"]
        tampered_settings = self._tampered_artifact("settings.json")
        with self.assertRaises(ZkError):
            run_verify(good, MODEL, settings_path=tampered_settings,
                       vk_path=self.setup_dir / "vk.key", srs_path=self.setup_dir / "kzg.srs")
        tampered_vk = self._tampered_artifact("vk.key")
        with self.assertRaises(ZkError):
            run_verify(good, MODEL, settings_path=self.setup_dir / "settings.json",
                       vk_path=tampered_vk, srs_path=self.setup_dir / "kzg.srs")
        tampered_srs = self._tampered_artifact("kzg.srs")
        with self.assertRaises(ZkError):
            run_verify(good, MODEL, settings_path=self.setup_dir / "settings.json",
                       vk_path=self.setup_dir / "vk.key", srs_path=tampered_srs)

    def test_wrong_model_fails(self):
        other = self._other_model()
        with self.assertRaises(ZkError):
            run_verify(self.credentials["normal"], other, **self._artifacts())

    def test_prove_rejects_model_setup_mismatch(self):
        with self.assertRaises(ZkError):
            run_prove(ROOT / "samples" / "normal.json", self._other_model(),
                      self.setup_dir, Path(self.temp_root.name) / "should-not-exist.json")

    def test_tampered_setup_is_not_reused(self):
        bad_setup = Path(self.temp_root.name) / "tampered-setup"
        bad_setup.mkdir()
        for name in ("settings.json", "circuit.ezkl", "vk.key", "kzg.srs", "pk.key", "manifest.json"):
            (bad_setup / name).write_bytes((self.setup_dir / name).read_bytes())
        with open(bad_setup / "pk.key", "r+b") as handle:
            handle.seek(4096)
            byte = handle.read(1)
            handle.seek(4096)
            handle.write(bytes([byte[0] ^ 0xFF]))
        with self.assertRaises(ZkError):
            run_prove(ROOT / "samples" / "normal.json", MODEL, bad_setup,
                      Path(self.temp_root.name) / "should-not-exist.json")

    def test_malformed_credentials_fail(self):
        for document in ("{not json", "", json.dumps({"format_version": "1.0"})):
            path = Path(self.temp_root.name) / "bad.json"
            path.write_text(document)
            with self.assertRaises(ZkError):
                run_verify(path, MODEL, **self._artifacts())

    def test_credential_with_features_key_rejected(self):
        credential = json.loads(self.credentials["normal"].read_text())
        credential["features"] = [0.125] * 6
        with self._temp_credential(credential) as path:
            with self.assertRaises(ZkError):
                run_verify(path, MODEL, **self._artifacts())

    def _tampered_artifact(self, name):
        source = self.setup_dir / name
        target = Path(self.temp_root.name) / f"tampered-{name}"
        target.write_bytes(source.read_bytes())
        with open(target, "r+b") as handle:
            handle.seek(1024)
            byte = handle.read(1)
            handle.seek(1024)
            handle.write(bytes([byte[0] ^ 0xFF]))
        return target

    def _other_model(self):
        path = Path(self.temp_root.name) / "other.onnx"
        if not path.exists():
            import numpy as np
            import onnx
            from onnx import TensorProto, helper, numpy_helper
            zeros = [numpy_helper.from_array(np.zeros((6, 4), dtype=np.float32), name="w1"),
                     numpy_helper.from_array(np.zeros(4, dtype=np.float32), name="b1"),
                     numpy_helper.from_array(np.zeros((4, 2), dtype=np.float32), name="w2"),
                     numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="b2")]
            nodes = [helper.make_node("MatMul", ["features", "w1"], ["h0"]),
                     helper.make_node("Add", ["h0", "b1"], ["h1"]),
                     helper.make_node("Relu", ["h1"], ["h2"]),
                     helper.make_node("MatMul", ["h2", "w2"], ["o0"]),
                     helper.make_node("Add", ["o0", "b2"], ["scores"])]
            graph = helper.make_graph(
                nodes, "other",
                [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 6])],
                [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2])], zeros)
            model = helper.make_model(graph, producer_name="other",
                                      opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8
            onnx.save(model, path)
        return path

    @contextmanager
    def _temp_credential(self, credential):
        path = Path(self.temp_root.name) / f"cred-{id(credential)}.json"
        path.write_text(json.dumps(credential))
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)


@unittest.skipUnless(os.environ.get("RUN_ZK_CLI_TESTS"), "set RUN_ZK_CLI_TESTS=1 for the slow CLI test")
class ZkCliTest(unittest.TestCase):
    def test_cli_loop_outputs_single_json(self):
        with tempfile.TemporaryDirectory(prefix="zkml-cli-") as directory:
            setup = Path(directory) / "setup"
            credential = Path(directory) / "cred.json"
            subprocess.run([sys.executable, "-m", "zkml_quality", "zk-setup",
                            "--model", str(MODEL), "--dir", str(setup)], check=True,
                           capture_output=True, text=True)
            subprocess.run([sys.executable, "-m", "zkml_quality", "zk-prove",
                            "--input", str(ROOT / "samples" / "normal.json"),
                            "--model", str(MODEL), "--setup-dir", str(setup),
                            "--credential", str(credential)], check=True,
                           capture_output=True, text=True)
            result = subprocess.run([sys.executable, "-m", "zkml_quality", "zk-verify",
                                     "--credential", str(credential), "--model", str(MODEL),
                                     "--settings", str(setup / "settings.json"),
                                     "--vk", str(setup / "vk.key"),
                                     "--srs", str(setup / "kzg.srs")],
                                    capture_output=True, text=True, check=True)
            payload = json.loads(result.stdout)  # exactly one JSON document
            self.assertEqual(payload["verified"], True)
            self.assertEqual(payload["quantized_scores"], [0.75, 0.25])
            self.assertEqual(payload["label"], "normal")
            self.assertEqual(payload["model_sha256"],
                             hashlib.sha256(MODEL.read_bytes()).hexdigest())
            self.assertEqual(set(payload), {"verified", "quantized_scores", "label", "model_sha256"})
            self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
