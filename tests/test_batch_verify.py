"""End-to-end tests for the offline zk-verify-batch command."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from zkml_quality.inference import ROOT
from zkml_quality.batch_verify import (
    MAX_ITEMS,
    BatchVerifyError,
    run_batch_verify,
)
from zkml_quality.registry import (
    enable_model,
    register_model,
    revoke_model,
)
from zkml_quality.zk import run_prove, run_setup, run_verify

MODEL = ROOT / "models" / "quality.onnx"
VERSION = "v1.0"


class BatchVerifyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = Path(tempfile.mkdtemp(prefix="zkbatch-"))
        setup_dir = cls.workspace / "setup"
        setup_info = run_setup(MODEL, setup_dir)
        cls.model_sha = setup_info["model_sha256"]

        cls.verifier_dir = cls.workspace / "verifier"
        cls.verifier_dir.mkdir()
        shutil.copy(setup_dir / "manifest.json", cls.verifier_dir / "manifest.json")
        shutil.copy(setup_dir / "settings.json", cls.verifier_dir / "settings.json")
        shutil.copy(setup_dir / "verification.key", cls.verifier_dir / "vk")
        shutil.copy(setup_dir / "srs", cls.verifier_dir / "srs")
        shutil.copy(MODEL, cls.verifier_dir / "model.onnx")

        cls.registry = cls.workspace / "registry.json"
        register_model(cls.registry, VERSION,
                       cls.verifier_dir / "manifest.json", cls.verifier_dir / "model.onnx")
        enable_model(cls.registry, VERSION)

        # Two honest credentials with different proven labels.
        cls.good_normal = cls._credential(setup_dir, "normal", [0.125] * 6)
        cls.good_inspect = cls._credential(
            setup_dir, "inspect", [0.75, 0.875, 0.75, 0.625, 0.75, 0.875])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    @classmethod
    def _credential(cls, setup_dir, name, values):
        input_path = cls.workspace / f"{name}.input.json"
        input_path.write_text(json.dumps({"features": list(values)}), encoding="utf-8")
        credential_path = cls.workspace / f"{name}.cred.json"
        run_prove(input_path, MODEL, setup_dir, credential_path)
        return credential_path

    def _trust_kwargs(self, **overrides):
        kwargs = dict(
            registry_path=self.registry,
            model_version=VERSION,
            manifest_path=self.verifier_dir / "manifest.json",
            model=self.verifier_dir / "model.onnx",
            settings_path=self.verifier_dir / "settings.json",
            vk_path=self.verifier_dir / "vk",
            srs_path=self.verifier_dir / "srs",
        )
        kwargs.update(overrides)
        return kwargs

    def _descriptor(self, items):
        path = self.workspace / f"batch-{next(self._counter)}.json"
        path.write_text(json.dumps({"items": items}), encoding="utf-8")
        return path

    def setUp(self):
        self._counter = iter(range(10000))

    def _run(self, items, **trust_overrides):
        descriptor = self._descriptor(items)
        return run_batch_verify(descriptor, **self._trust_kwargs(**trust_overrides))

    def _write_bad_credential(self, name, text):
        path = self.workspace / name
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
        path.write_text(text, encoding="utf-8")
        return path

    # ---- happy path / per-item results -------------------------------------

    def test_mixed_batch_keeps_order_and_isolates_rejections(self):
        tampered_proof = json.loads(self.good_normal.read_text(encoding="utf-8"))
        tampered_proof["proof"]["proof"][40] ^= 0xFF
        proof_path = self._write_bad_credential("bad-proof.json", tampered_proof)

        # Public-output tamper that stays internally consistent, so it passes
        # credential parsing but cannot match the proven instances.
        tampered_score = json.loads(self.good_normal.read_text(encoding="utf-8"))
        tampered_score["public_output"]["quantized_scores"][0] += 1
        scale = tampered_score["public_output"]["output_scale"]
        tampered_score["public_output"]["scores_fixed_point"] = [
            value / (1 << scale)
            for value in tampered_score["public_output"]["quantized_scores"]]
        score_path = self._write_bad_credential("bad-score.json", tampered_score)

        tampered_label = json.loads(self.good_normal.read_text(encoding="utf-8"))
        tampered_label["public_output"]["label"] = "inspect"
        label_path = self._write_bad_credential("bad-label.json", tampered_label)

        broken_json = self._write_bad_credential("broken.json", "{not json")
        wrong_kind = self._write_bad_credential(
            "wrong-kind.json", {"kind": "something-else", "format_version": 1})
        missing = self.workspace / "does-not-exist.json"

        items = [
            {"id": "ok-1", "credential": str(self.good_normal)},
            {"id": "gone", "credential": str(missing)},
            {"id": "ok-2", "credential": str(self.good_inspect)},
            {"id": "bad-proof", "credential": str(proof_path)},
            {"id": "bad-score", "credential": str(score_path)},
            {"id": "bad-label", "credential": str(label_path)},
            {"id": "bad-json", "credential": str(broken_json)},
            {"id": "bad-kind", "credential": str(wrong_kind)},
            # A directory is not a readable credential file.
            {"id": "is-dir", "credential": str(self.workspace)},
        ]
        result = self._run(items)

        self.assertEqual(result["total"], 9)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["rejected"], 7)
        self.assertEqual([row["id"] for row in result["results"]],
                         [item["id"] for item in items])

        accepted = {row["id"]: row for row in result["results"] if row["accepted"]}
        single = run_verify(self.good_inspect, **self._trust_kwargs())
        self.assertEqual(set(accepted["ok-2"]),
                         {"id", "accepted", "verified", "model_sha256",
                          "quantized_scores", "scores_fixed_point", "label"})
        self.assertTrue(accepted["ok-2"]["accepted"])
        self.assertTrue(accepted["ok-2"]["verified"])
        self.assertEqual(accepted["ok-2"]["label"], "inspect")
        for field in ("model_sha256", "quantized_scores",
                      "scores_fixed_point", "label"):
            self.assertEqual(accepted["ok-2"][field], single[field])
        self.assertEqual(accepted["ok-1"]["label"], "normal")

        codes = {row["id"]: row["error"]["code"]
                 for row in result["results"] if not row["accepted"]}
        self.assertEqual(codes, {
            "gone": "credential_missing",
            "is-dir": "credential_missing",
            "bad-proof": "verification_failed",
            "bad-score": "verification_failed",
            "bad-label": "verification_failed",
            "bad-json": "credential_invalid",
            "bad-kind": "credential_invalid",
        })
        for row in result["results"]:
            if not row["accepted"]:
                self.assertEqual(set(row), {"id", "accepted", "error"})
                self.assertEqual(
                    set(row["error"]), {"code", "message"})
                self.assertIsInstance(row["error"]["message"], str)
                self.assertTrue(row["error"]["message"])

        # The honest credentials are untouched and still verify singly.
        self.assertTrue(run_verify(self.good_normal, **self._trust_kwargs())["verified"])

    def test_one_bad_item_does_not_block_the_rest(self):
        result = self._run([
            {"id": "a", "credential": str(self.workspace / "missing-a.json")},
            {"id": "b", "credential": str(self.good_normal)},
            {"id": "c", "credential": str(self.workspace / "missing-c.json")},
        ])
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["rejected"], 2)
        self.assertEqual(
            [row["accepted"] for row in result["results"]], [False, True, False])

    # ---- descriptor validation: the whole batch is refused ------------------

    def test_malformed_descriptors_reject_the_whole_batch(self):
        def reject(raw, *, label):
            path = self.workspace / f"desc-{label}.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(BatchVerifyError) as caught:
                run_batch_verify(path, **self._trust_kwargs())
            self.assertEqual(caught.exception.code, "invalid_batch")

        good_item = {"id": "x", "credential": str(self.good_normal)}

        reject(json.dumps([]), label="array-toplevel")
        reject(json.dumps({}), label="empty-object")
        reject(json.dumps({"items": []}), label="empty-items")
        reject(json.dumps({"items": {}}), label="items-object")
        reject(json.dumps({"items": good_item}), label="items-not-array")
        reject(json.dumps({"items": [good_item], "extra": 1}), label="extra-toplevel")
        reject(json.dumps({"items": [[]]}), label="item-array")
        reject(json.dumps({"items": ["x"]}), label="item-string")
        reject(json.dumps({"items": [{}]}), label="item-empty")
        reject(json.dumps(
            {"items": [{"id": "x", "credential": str(self.good_normal),
                        "extra": 1}]}), label="extra-item-field")
        reject(json.dumps(
            {"items": [{"id": "x"}]}), label="missing-credential")
        reject(json.dumps(
            {"items": [{"credential": str(self.good_normal)}]}), label="missing-id")
        for index, bad_id in enumerate(
                ("", "bad id!", "a/b", "a:b", "id#1", 123, None, True)):
            reject(json.dumps(
                {"items": [{"id": bad_id, "credential": str(self.good_normal)}]}),
                label=f"id-{index}")
        reject(json.dumps(
            {"items": [{"id": "x", "credential": ""}]}), label="empty-credential")
        reject(json.dumps(
            {"items": [{"id": "x", "credential": 3}]}), label="numeric-credential")
        reject(json.dumps(
            {"items": [good_item, good_item]}), label="duplicate-id")
        reject("{not json", label="syntax")
        # Raw text with explicit duplicate keys (json.dumps would collapse them).
        one_item = json.dumps([good_item])
        reject('{"items": ' + one_item + ', "items": ' + one_item + "}",
               label="dup-toplevel-key")
        reject('{"items": [{"id": "x", "id": "y", '
               f'"credential": {json.dumps(str(self.good_normal))}' + "]}",
               label="dup-item-key")

        with self.assertRaises(BatchVerifyError) as caught:
            run_batch_verify(self.workspace / "no-descriptor.json",
                             **self._trust_kwargs())
        self.assertEqual(caught.exception.code, "invalid_batch")

    def test_item_limit_is_256(self):
        # 256 items is admitted; every credential is missing, so no EZKL work
        # happens and all 256 are individually rejected without aborting.
        items = [{"id": f"id-{index}",
                  "credential": str(self.workspace / f"missing-{index}.json")}
                 for index in range(MAX_ITEMS)]
        result = self._run(items)
        self.assertEqual(result["total"], MAX_ITEMS)
        self.assertEqual(result["accepted"], 0)
        self.assertEqual(result["rejected"], MAX_ITEMS)
        self.assertTrue(all(
            row["error"]["code"] == "credential_missing" for row in result["results"]))

        # 257 items refuses the whole batch before any verification.
        too_many = items + [{"id": "one-more",
                             "credential": str(self.good_normal)}]
        with self.assertRaises(BatchVerifyError) as caught:
            self._run(too_many)
        self.assertEqual(caught.exception.code, "invalid_batch")

    # ---- trust preflight aborts everything ----------------------------------

    def test_trust_failure_aborts_the_whole_batch(self):
        items = [{"id": "a", "credential": str(self.good_normal)},
                 {"id": "b", "credential": str(self.good_inspect)}]

        with self.assertRaises(BatchVerifyError) as caught:
            self._run(items, model_version="not-registered")
        self.assertEqual(caught.exception.code, "trust_check_failed")

        disabled = self.workspace / "registry-disabled.json"
        register_model(disabled, "d1",
                       self.verifier_dir / "manifest.json",
                       self.verifier_dir / "model.onnx")
        with self.assertRaises(BatchVerifyError) as caught:
            self._run(items, registry_path=disabled, model_version="d1")
        self.assertEqual(caught.exception.code, "trust_check_failed")

        revoked = self.workspace / "registry-revoked.json"
        register_model(revoked, "r1",
                       self.verifier_dir / "manifest.json",
                       self.verifier_dir / "model.onnx")
        enable_model(revoked, "r1")
        revoke_model(revoked, "r1")
        with self.assertRaises(BatchVerifyError) as caught:
            self._run(items, registry_path=revoked, model_version="r1")
        self.assertEqual(caught.exception.code, "trust_check_failed")

        with self.assertRaises(BatchVerifyError) as caught:
            self._run(items, registry_path=self.workspace / "no-registry.json")
        self.assertEqual(caught.exception.code, "trust_check_failed")

        corrupt = self.workspace / "registry-corrupt.json"
        corrupt.write_text("{not json", encoding="utf-8")
        with self.assertRaises(BatchVerifyError) as caught:
            self._run(items, registry_path=corrupt)
        self.assertEqual(caught.exception.code, "trust_check_failed")

        # A missing trust material file likewise aborts before any item runs.
        with self.assertRaises(BatchVerifyError) as caught:
            self._run(items, vk_path=self.workspace / "missing-vk")
        self.assertEqual(caught.exception.code, "trust_check_failed")

    def test_trust_check_runs_before_credentials_are_read(self):
        # Even a descriptor full of missing/invalid credentials is aborted
        # wholesale by the trust gate rather than processed.
        items = [{"id": "a", "credential": str(self.workspace / "nope.json")}]
        with self.assertRaises(BatchVerifyError):
            self._run(items, model_version="not-registered")

    # ---- privacy -------------------------------------------------------------

    def test_output_never_leaks_paths_features_or_proofs(self):
        tampered = json.loads(self.good_normal.read_text(encoding="utf-8"))
        tampered["proof"]["proof"][40] ^= 0xFF
        bad = self._write_bad_credential(
            "leak-bad.json", tampered)
        # Give the credential a path containing a distinctive scratch marker.
        result = self._run([
            {"id": "ok", "credential": str(self.good_normal)},
            {"id": "bad", "credential": str(bad)},
            {"id": "missing",
             "credential": str(self.workspace / "deep" / "secret-cred.json")},
        ])
        rendered = json.dumps(result)
        self.assertNotIn(str(self.workspace), rendered)
        self.assertNotIn("features", rendered)
        self.assertNotIn("input_data", rendered)
        self.assertNotIn("proof", rendered)
        # The only credential-ish text allowed is the fixed safe messages; no
        # credential path may appear anywhere.
        self.assertNotIn(str(bad), rendered)
        self.assertNotIn("secret-cred", rendered)

    # ---- CLI -----------------------------------------------------------------

    def _cli(self, descriptor, *extra_args):
        return subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify-batch",
             "--file", str(descriptor),
             "--registry", str(self.registry), "--model-version", VERSION,
             "--manifest", str(self.verifier_dir / "manifest.json"),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs"), *extra_args],
            cwd=str(ROOT), capture_output=True, text=True)

    def test_cli_success_with_rejections_exits_zero_single_json(self):
        descriptor = self._descriptor([
            {"id": "ok", "credential": str(self.good_normal)},
            {"id": "missing", "credential": str(self.workspace / "gone.json")},
        ])
        proc = self._cli(descriptor)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        result = json.loads(proc.stdout)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual([row["id"] for row in result["results"]],
                         ["ok", "missing"])
        self.assertEqual(
            result["results"][1]["error"]["code"], "credential_missing")
        # Exactly one JSON object on stdout.
        self.assertEqual(proc.stdout.count("{"), proc.stdout.count("}"))

    def test_cli_invalid_descriptor_is_nonzero_empty_stdout_safe_stderr(self):
        descriptor = self.workspace / "cli-bad-descriptor.json"
        descriptor.write_text(json.dumps({"items": []}), encoding="utf-8")
        proc = self._cli(descriptor)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        error = json.loads(proc.stderr)
        self.assertEqual(set(error), {"error"})
        self.assertEqual(error["error"]["code"], "invalid_batch")
        self.assertNotIn(str(self.workspace), proc.stderr)

    def test_cli_trust_failure_is_nonzero_empty_stdout_safe_stderr(self):
        descriptor = self._descriptor([
            {"id": "ok", "credential": str(self.good_normal)}])
        proc = self._cli(descriptor, "--model-version", "not-registered")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        error = json.loads(proc.stderr)
        self.assertEqual(error["error"]["code"], "trust_check_failed")
        self.assertNotIn(str(self.workspace), proc.stderr)

    def test_cli_missing_file_argument_fails(self):
        proc = subprocess.run(
            [sys.executable, "-m", "zkml_quality", "zk-verify-batch",
             "--registry", str(self.registry), "--model-version", VERSION,
             "--manifest", str(self.verifier_dir / "manifest.json"),
             "--model", str(self.verifier_dir / "model.onnx"),
             "--settings", str(self.verifier_dir / "settings.json"),
             "--vk", str(self.verifier_dir / "vk"),
             "--srs", str(self.verifier_dir / "srs")],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("--file", proc.stderr)


if __name__ == "__main__":
    unittest.main()
