"""Real EZKL 23.0.5 CPU zero-knowledge proof loop for the quality model.

Three stages:

* ``run_setup`` compiles an ONNX model into a circuit and produces settings,
  proving key, verification key and SRS (offline, CPU).
* ``run_prove`` consumes a private six-feature input, generates the witness and
  a real proof, and writes a self-contained JSON credential. Only the two
  quantized output scores are public; the features stay private.
* ``run_verify`` checks a credential against verifier-supplied model, settings,
  VK and SRS. It needs neither the original input nor the proving key nor any
  network access.

The ordinary ONNX floating-point scores are never presented as proven values:
the credential only carries the quantized scores that are the circuit's public
instances.
"""
import contextlib
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import ezkl

from .inference import validate_features

FORMAT_VERSION = 1
CREDENTIAL_KIND = "zk-quality-credential"
ARTIFACT_NAMES = {
    "compiled": "compiled.ezkl",
    "settings": "settings.json",
    "pk": "proving.key",
    "vk": "verification.key",
    "srs": "srs",
}
MANIFEST_NAME = "manifest.json"

# Labels follow the existing inference convention: index 0 is "normal",
# index 1 is "inspect", and a tie resolves to "normal".
LABELS = ("normal", "inspect")


class ZkError(Exception):
    """Raised for any invalid input, artifact mismatch or EZKL failure."""


def ezkl_version():
    try:
        return importlib.metadata.version("ezkl")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise ZkError(f"{description} not found: {path}")
    return path


@contextlib.contextmanager
def _quiet_stderr():
    """Redirect fd-level stderr (EZKL writes via Rust, bypassing Python).

    Captured output is attached to the raised error so EZKL's own diagnostic
    is still surfaced on failure.
    """
    captured = tempfile.TemporaryFile(mode="w+b")
    saved = os.dup(2)
    try:
        os.dup2(captured.fileno(), 2)
        yield
    except Exception as error:
        os.dup2(saved, 2)
        captured.seek(0)
        tail = captured.read().decode("utf-8", "replace").strip().splitlines()
        detail = tail[-1].strip() if tail else str(error)
        if isinstance(error, ZkError):
            raise
        raise ZkError(f"EZKL failed: {detail}") from error
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        captured.close()


def _calibration_rows():
    """Deterministic domain-covering calibration data.

    The network is a stack of linear layers and ReLUs, so per-output extrema
    over the six-dimensional [0, 1] input cube occur at the 2**6 binary
    vertices. The two bundled sample vectors are added for good measure.
    """
    rows = [[float((i >> bit) & 1) for bit in range(6)] for i in range(1 << 6)]
    rows.append([0.125] * 6)
    rows.append([0.75, 0.875, 0.75, 0.625, 0.75, 0.875])
    return rows


def run_setup(model, setup_dir):
    """Compile the model and generate settings, SRS, proving and verifying keys."""
    model = _require_file(model, "ONNX model")
    setup_dir = Path(setup_dir)
    if setup_dir.exists() and not setup_dir.is_dir():
        raise ZkError(f"setup destination is not a directory: {setup_dir}")
    if setup_dir.is_dir() and any(setup_dir.iterdir()):
        raise ZkError(f"setup directory is not empty, refusing to mix artifacts: {setup_dir}")
    setup_dir.mkdir(parents=True, exist_ok=True)

    model_sha = _sha256(model)
    partials = []
    try:
        with tempfile.TemporaryDirectory(prefix="zksetup-") as scratch:
            scratch = Path(scratch)
            calib_path = scratch / "calibration.json"
            calib_path.write_text(json.dumps({"input_data": _calibration_rows()}), encoding="utf-8")

            targets = {key: setup_dir / name for key, name in ARTIFACT_NAMES.items()}
            parts = {key: setup_dir / f".{name}.part" for key, name in ARTIFACT_NAMES.items()}
            partials = list(parts.values())

            run_args = ezkl.PyRunArgs()
            run_args.input_visibility = "private"
            run_args.output_visibility = "public"
            run_args.param_visibility = "fixed"

            settings_part = parts["settings"]
            with _quiet_stderr():
                if not ezkl.gen_settings(str(model), str(settings_part), run_args):
                    raise ZkError("gen_settings returned false")
                if not ezkl.calibrate_settings(
                    str(calib_path), str(model), str(settings_part),
                    "resources", 0.1, None, [1], None,
                ):
                    raise ZkError("calibrate_settings returned false")
                if not ezkl.compile_circuit(str(model), str(parts["compiled"]), str(settings_part)):
                    raise ZkError("compile_circuit returned false")

            settings_doc = json.loads(settings_part.read_text(encoding="utf-8"))
            try:
                logrows = int(settings_doc["run_args"]["logrows"])
                output_scale = int(settings_doc["model_output_scales"][0])
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ZkError(f"compiled settings are missing logrows/output scale: {error}") from error
            if output_scale <= 0 or not 1 <= logrows <= 30:
                raise ZkError("compiled settings contain implausible scales/logrows")

            with _quiet_stderr():
                ezkl.gen_srs(str(parts["srs"]), logrows)
                if not ezkl.setup(
                    str(parts["compiled"]), str(parts["vk"]), str(parts["pk"]),
                    str(parts["srs"]), None, False,
                ):
                    raise ZkError("key setup returned false")

            for key, target in targets.items():
                os.replace(parts[key], target)
                partials.append(target)

            manifest = {
                "format_version": FORMAT_VERSION,
                "kind": "zk-quality-setup",
                "ezkl_version": ezkl_version(),
                "model_sha256": model_sha,
                "logrows": logrows,
                "output_scale": output_scale,
                "artifacts": {key: _sha256(targets[key]) for key in targets},
            }
            manifest_part = setup_dir / f".{MANIFEST_NAME}.part"
            manifest_part.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            partials.append(manifest_part)
            os.replace(manifest_part, setup_dir / MANIFEST_NAME)
            partials.append(setup_dir / MANIFEST_NAME)
            partials = []
    except Exception:
        for leftover in partials:
            with contextlib.suppress(OSError):
                leftover.unlink()
        raise

    return {"setup_dir": str(setup_dir), "model_sha256": model_sha, "logrows": logrows,
            "output_scale": output_scale}


def _load_manifest(setup_dir):
    manifest_path = _require_file(setup_dir / MANIFEST_NAME, "setup manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ZkError(f"invalid setup manifest JSON: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("kind") != "zk-quality-setup" \
            or manifest.get("format_version") != FORMAT_VERSION:
        raise ZkError("setup manifest is unsupported or was not produced by zk-setup")
    if manifest.get("ezkl_version") != ezkl_version():
        raise ZkError(
            f"setup was produced with EZKL {manifest.get('ezkl_version')}, "
            f"installed EZKL is {ezkl_version()}")
    return manifest


def _artifact_paths(setup_dir):
    return {key: _require_file(setup_dir / name, f"setup artifact '{name}'")
            for key, name in ARTIFACT_NAMES.items()}


def _check_artifacts(manifest, paths):
    """Verify every setup artifact on disk still matches the setup manifest."""
    digests = manifest.get("artifacts")
    if not isinstance(digests, dict):
        raise ZkError("setup manifest does not record artifact digests")
    for key, path in paths.items():
        if _sha256(path) != digests.get(key):
            raise ZkError(
                f"setup artifact '{ARTIFACT_NAMES[key]}' does not match the setup manifest; "
                "rerun zk-setup instead of reusing mismatched files")


def _decode_instances(instances, scale):
    if not isinstance(instances, list) or len(instances) != 1 or not isinstance(instances[0], list):
        raise ZkError("proof does not expose exactly one public output row")
    row = instances[0]
    if len(row) != 2 or not all(isinstance(value, str) for value in row):
        raise ZkError("proof must expose exactly two public quantized scores")
    scores = []
    for felt in row:
        try:
            scores.append(int(ezkl.felt_to_int(felt)))
        except (TypeError, ValueError, RuntimeError) as error:
            raise ZkError(f"public score '{felt}' is not a valid field element: {error}") from error
    if not all(-(1 << 31) < score < (1 << 31) for score in scores):
        raise ZkError("public quantized scores fall outside the plausible range")
    label = LABELS[0] if scores[0] >= scores[1] else LABELS[1]
    fixed_point = [score / (1 << scale) for score in scores]
    return scores, fixed_point, label


def _read_proof_document(path):
    try:
        raw = Path(path).read_bytes()
        proof_doc = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ZkError(f"cannot read proof document: {error}") from error
    if not isinstance(proof_doc, dict) or not isinstance(proof_doc.get("instances"), list):
        raise ZkError("proof document is malformed")
    return proof_doc


def run_prove(input_path, model, setup_dir, credential_path):
    """Generate witness and a real proof for one private feature vector."""
    input_path = _require_file(input_path, "input file")
    model = _require_file(model, "ONNX model")
    setup_dir = Path(setup_dir)
    if not setup_dir.is_dir():
        raise ZkError(f"setup directory not found: {setup_dir}")
    credential_path = Path(credential_path)
    if credential_path.is_dir():
        raise ZkError(f"credential path is a directory: {credential_path}")
    if not credential_path.resolve().parent.is_dir():
        raise ZkError(f"credential output directory does not exist: {credential_path.parent}")

    try:
        document = json.loads(input_path.read_text(encoding="utf-8"),
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValueError(f"invalid JSON constant {value}")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ZkError(f"invalid input JSON ({input_path}): {error}") from error
    try:
        features = validate_features(document)  # identical six-dimension contract as `infer`
    except ValueError as error:
        raise ZkError(str(error)) from error

    manifest = _load_manifest(setup_dir)
    model_sha = _sha256(model)
    if model_sha != manifest.get("model_sha256"):
        raise ZkError("model does not match the model this setup was built for")
    paths = _artifact_paths(setup_dir)
    _check_artifacts(manifest, paths)
    scale = int(manifest["output_scale"])

    with tempfile.TemporaryDirectory(prefix="zkprove-") as scratch:
        scratch = Path(scratch)
        ezkl_input = scratch / "witness-input.json"
        witness_path = scratch / "witness.json"
        proof_path = scratch / "proof.pf"
        ezkl_input.write_text(json.dumps({"input_data": [features]}), encoding="utf-8")
        with _quiet_stderr():
            witness = ezkl.gen_witness(
                str(ezkl_input), str(paths["compiled"]), str(witness_path), None, None)
            if not isinstance(witness, dict) or "outputs" not in witness:
                raise ZkError("witness generation produced no public outputs")
            proof_result = ezkl.prove(str(witness_path), str(paths["compiled"]),
                                      str(paths["pk"]), str(proof_path), str(paths["srs"]))
            if not proof_path.is_file() or not isinstance(proof_result, dict):
                raise ZkError("proving did not produce a proof")
            # Self-check before issuing a credential; requires local artifacts only.
            if ezkl.verify(str(proof_path), str(paths["settings"]), str(paths["vk"]),
                           str(paths["srs"]), False) is not True:
                raise ZkError("freshly generated proof failed local verification")

        proof_doc = _read_proof_document(proof_path)
        scores, fixed_point, label = _decode_instances(proof_doc["instances"], scale)

        # The witness also reports the decoded outputs; they must agree with the
        # public instances that actually went into the proof.
        witness_scores, _, witness_label = _decode_instances(witness["outputs"], scale)
        if witness_scores != scores or witness_label != label:
            raise ZkError("witness outputs and proof instances disagree")

    credential = {
        "format_version": FORMAT_VERSION,
        "kind": CREDENTIAL_KIND,
        "ezkl_version": ezkl_version(),
        "model_sha256": model_sha,
        "verification_artifacts": {
            "settings_sha256": _sha256(paths["settings"]),
            "vk_sha256": _sha256(paths["vk"]),
            "srs_sha256": _sha256(paths["srs"]),
        },
        "public_output": {
            "output_scale": scale,
            "quantized_scores": scores,
            "scores_fixed_point": fixed_point,
            "label": label,
        },
        "proof": proof_doc,
    }
    _write_credential(credential, credential_path)
    return {"credential": str(credential_path), "model_sha256": model_sha,
            "quantized_scores": scores, "label": label}


def _write_credential(credential, credential_path):
    credential_path = Path(credential_path)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=credential_path.resolve().parent,
        prefix=".credential-", suffix=".tmp", delete=False)
    try:
        json.dump(credential, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
        handle.close()
        os.replace(handle.name, credential_path)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def _load_credential(credential_path):
    credential_path = _require_file(credential_path, "credential")
    try:
        raw = Path(credential_path).read_text(encoding="utf-8")
        credential = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {value}")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ZkError(f"invalid credential JSON: {error}") from error
    if not isinstance(credential, dict):
        raise ZkError("credential must be a JSON object")
    if credential.get("kind") != CREDENTIAL_KIND or credential.get("format_version") != FORMAT_VERSION:
        raise ZkError("unsupported credential format")
    for field in ("ezkl_version", "model_sha256", "verification_artifacts", "public_output", "proof"):
        if field not in credential:
            raise ZkError(f"credential is missing '{field}'")
    if credential["ezkl_version"] != ezkl_version():
        raise ZkError(
            f"credential was produced with EZKL {credential['ezkl_version']}, "
            f"verifier uses EZKL {ezkl_version()}")
    if not isinstance(credential["model_sha256"], str) \
            or len(credential["model_sha256"]) != 64:
        raise ZkError("credential model digest is invalid")
    # A credential must never carry the private input.
    if "features" in credential or "input_data" in credential:
        raise ZkError("credential illegally contains private input data")
    artifacts = credential["verification_artifacts"]
    if not isinstance(artifacts, dict) or any(
            not isinstance(artifacts.get(key), str) or len(artifacts[key]) != 64
            for key in ("settings_sha256", "vk_sha256", "srs_sha256")):
        raise ZkError("credential verification-artifact digests are invalid")
    public = credential["public_output"]
    if not isinstance(public, dict) or not isinstance(public.get("quantized_scores"), list) \
            or len(public["quantized_scores"]) != 2 \
            or not all(isinstance(v, int) and not isinstance(v, bool)
                       for v in public["quantized_scores"]) \
            or public.get("label") not in LABELS \
            or not isinstance(public.get("output_scale"), int) or public["output_scale"] <= 0:
        raise ZkError("credential public output is invalid")
    expected_fixed = [score / (1 << public["output_scale"])
                      for score in public["quantized_scores"]]
    if public.get("scores_fixed_point") != expected_fixed:
        raise ZkError("credential fixed-point scores do not match the quantized scores")
    if not isinstance(credential["proof"], dict) \
            or not isinstance(credential["proof"].get("instances"), list):
        raise ZkError("credential proof payload is invalid")
    return credential


def run_verify(credential_path, model, settings_path, vk_path, srs_path):
    """Verify a credential offline against verifier-chosen public artifacts.

    The credential's embedded model hash and parameters are never trusted:
    the model digest is recomputed from the verifier's own ONNX file, artifact
    digests are recomputed from the verifier's settings/VK/SRS, and the final
    decision comes from EZKL cryptographic verification.
    """
    credential = _load_credential(credential_path)
    model = _require_file(model, "ONNX model")
    settings_path = _require_file(settings_path, "settings file")
    vk_path = _require_file(vk_path, "verification key")
    srs_path = _require_file(srs_path, "SRS file")

    model_sha = _sha256(model)
    if model_sha != credential["model_sha256"]:
        raise ZkError("model digest does not match the credential")
    claimed = credential["verification_artifacts"]
    supplied = {
        "settings_sha256": (_sha256(settings_path), settings_path),
        "vk_sha256": (_sha256(vk_path), vk_path),
        "srs_sha256": (_sha256(srs_path), srs_path),
    }
    for key, (actual, path) in supplied.items():
        if actual != claimed[key]:
            raise ZkError(f"verification artifact mismatch for {key}: {path}")

    try:
        settings_doc = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        settings_scale = int(settings_doc["model_output_scales"][0])
        settings_version = str(settings_doc["version"])
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise ZkError(f"settings file is unreadable: {error}") from error
    if settings_version != credential["ezkl_version"]:
        raise ZkError(
            f"settings were produced with EZKL {settings_version}, "
            f"credential with EZKL {credential['ezkl_version']}")
    if settings_scale != credential["public_output"]["output_scale"]:
        raise ZkError("settings output scale does not match the credential")

    with tempfile.TemporaryDirectory(prefix="zkverify-") as scratch:
        proof_path = Path(scratch) / "proof.pf"
        proof_path.write_text(json.dumps(credential["proof"]), encoding="utf-8")
        with _quiet_stderr():
            accepted = ezkl.verify(str(proof_path), str(settings_path), str(vk_path),
                                   str(srs_path), False)
        if accepted is not True:
            raise ZkError("proof verification failed")

    scores, fixed_point, label = _decode_instances(
        credential["proof"]["instances"], settings_scale)
    if scores != credential["public_output"]["quantized_scores"]:
        raise ZkError("credential public scores do not match the proven instances")
    if label != credential["public_output"]["label"]:
        raise ZkError("credential label does not match the proven instances")

    return {"verified": True, "model_sha256": model_sha,
            "quantized_scores": scores, "scores_fixed_point": fixed_point, "label": label}
