"""Real EZKL 23.0.5 zero-knowledge proof pipeline for the quality model.

Three stages, exposed via the CLI: ``zk-setup`` compiles the circuit and
creates the proving/verifying artifacts; ``zk-prove`` executes the witness
and produces a real KZG proof over private features; ``zk-verify`` checks
that proof with verifier-chosen artifacts and no private input, proving key,
or network access.

The only values ever claimed as proven are the circuit's public outputs,
decoded from the verified proof's field elements. Ordinary ONNX runtime
scores are never placed in a credential.
"""
import base64
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import ezkl

from .inference import extract_features

CREDENTIAL_FORMAT = "1.0"
REQUIRED_EZKL_VERSION = "23.0.5"

CIRCUIT_NAME = "circuit.ezkl"
SETTINGS_NAME = "settings.json"
PK_NAME = "pk.key"
VK_NAME = "vk.key"
SRS_NAME = "kzg.srs"
MANIFEST_NAME = "manifest.json"

CREDENTIAL_KEYS = {
    "format_version",
    "ezkl_version",
    "model_sha256",
    "artifacts",
    "proof",
    "public_output",
}


class ZkError(Exception):
    """Raised for any user-facing setup/prove/verify failure."""


def _ezkl(description, function, *args, **kwargs):
    """Run an EZKL call, converting every failure into a clear ZkError."""
    try:
        return function(*args, **kwargs)
    except Exception as exc:  # ezkl raises RuntimeError on every failure
        raise ZkError(f"EZKL failed while {description}: {exc}") from exc


def _ezkl_version():
    try:
        return version("ezkl")
    except PackageNotFoundError:
        return getattr(ezkl, "__version__", "unknown")


def _require_file(path, what):
    path = Path(path)
    if not path.is_file():
        raise ZkError(f"{what} not found at {path}")
    return path


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path, what):
    path = _require_file(path, what)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ZkError(f"cannot read {what} {path}: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ZkError(f"{what} {path} is not valid JSON: {exc}") from exc


def _write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


@contextmanager
def _workspace(prefix):
    """A private scratch directory; witness/proof files are never reused."""
    directory = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _run_args():
    args = ezkl.PyRunArgs()
    args.input_visibility = "Private"
    args.output_visibility = "Public"
    args.param_visibility = "Fixed"
    return args


def _output_scale(settings):
    scales = settings.get("model_output_scales")
    if not isinstance(scales, list) or len(scales) != 1 \
            or not isinstance(scales[0], int) or isinstance(scales[0], bool) or scales[0] <= 0:
        raise ZkError("settings do not declare a single positive output scale")
    return scales[0]


def _decode_instances(instances, scale):
    """Turn the proof's two public field elements into integers and scores."""
    if not isinstance(instances, list) or len(instances) != 1 or not isinstance(instances[0], list):
        raise ZkError("proof must expose exactly one public output row")
    row = instances[0]
    if len(row) != 2 or not all(isinstance(value, str) for value in row):
        raise ZkError("the circuit must expose exactly two public scores")
    integers = []
    for felt in row:
        try:
            integers.append(int(ezkl.felt_to_int(felt)))
        except Exception as exc:
            raise ZkError(f"cannot decode public output field element {felt!r}: {exc}") from exc
    scores = [value / (1 << scale) for value in integers]
    label = "normal" if integers[0] >= integers[1] else "inspect"  # ties -> normal
    return integers, scores, label


def run_setup(model_path, output_dir):
    """Compile the ONNX circuit and generate settings, SRS, pk and vk."""
    if _ezkl_version() != REQUIRED_EZKL_VERSION:
        raise ZkError(f"ezkl {REQUIRED_EZKL_VERSION} is required (found {_ezkl_version()})")
    model_path = _require_file(model_path, "ONNX model")
    model_bytes = model_path.read_bytes()
    if not model_bytes:
        raise ZkError(f"ONNX model {model_path} is empty")
    model_hash = hashlib.sha256(model_bytes).hexdigest()

    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ZkError(f"setup target {output_dir} exists and is not a directory")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ZkError(f"setup directory {output_dir} must be empty to avoid reusing mismatched artifacts")

    with _workspace("zkml-setup-") as stage:
        staged_model = stage / "model.onnx"
        staged_model.write_bytes(model_bytes)
        settings_path = stage / SETTINGS_NAME
        compiled_path = stage / CIRCUIT_NAME
        srs_path = stage / SRS_NAME
        vk_path = stage / VK_NAME
        pk_path = stage / PK_NAME
        calibration_path = stage / "calibration.json"
        # The network has non-negative weights, so the [0, 1] domain corners
        # bound every interior activation and output magnitude.
        calibration_path.write_text(
            json.dumps({"input_data": [[0.0] * 6, [1.0] * 6], "output_data": None}),
            encoding="utf-8",
        )

        _ezkl("generating settings", ezkl.gen_settings,
              model=str(staged_model), output=str(settings_path), py_run_args=_run_args())
        _ezkl("calibrating settings", ezkl.calibrate_settings,
              data=str(calibration_path), model=str(staged_model),
              settings=str(settings_path), target="resources")
        settings = _read_json(settings_path, "generated settings")
        logrows = settings.get("run_args", {}).get("logrows")
        scale = _output_scale(settings)
        if not isinstance(logrows, int) or logrows <= 0:
            raise ZkError("generated settings have no valid logrows value")

        _ezkl("compiling the circuit", ezkl.compile_circuit,
              model=str(staged_model), compiled_circuit=str(compiled_path),
              settings_path=str(settings_path))
        _ezkl("generating the SRS", ezkl.gen_srs, str(srs_path), logrows)
        _ezkl("running setup", ezkl.setup,
              model=str(compiled_path), vk_path=str(vk_path),
              pk_path=str(pk_path), srs_path=str(srs_path))

        for path in (settings_path, compiled_path, srs_path, vk_path, pk_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise ZkError(f"EZKL did not produce expected artifact {path.name}")

        artifacts = {
            "compiled_circuit_sha256": _sha256_file(compiled_path),
            "settings_sha256": _sha256_file(settings_path),
            "pk_sha256": _sha256_file(pk_path),
            "vk_sha256": _sha256_file(vk_path),
            "srs_sha256": _sha256_file(srs_path),
        }
        manifest = {
            "format_version": CREDENTIAL_FORMAT,
            "ezkl_version": REQUIRED_EZKL_VERSION,
            "model_sha256": model_hash,
            "logrows": logrows,
            "output_scale": scale,
            "artifacts": artifacts,
        }
        (stage / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            for name in (SETTINGS_NAME, CIRCUIT_NAME, SRS_NAME, VK_NAME, PK_NAME, MANIFEST_NAME):
                shutil.move(str(stage / name), str(output_dir / name))
        except BaseException:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise

    return {"dir": str(output_dir), "model_sha256": model_hash, "logrows": logrows,
            "output_scale": scale, "artifacts": artifacts}


def _load_setup(setup_dir):
    setup_dir = _require_file(setup_dir / MANIFEST_NAME, "setup manifest").parent
    manifest = _read_json(setup_dir / MANIFEST_NAME, "setup manifest")
    paths = {}
    for key, name in (("compiled_circuit", CIRCUIT_NAME), ("settings", SETTINGS_NAME),
                      ("pk", PK_NAME), ("vk", VK_NAME), ("srs", SRS_NAME)):
        path = _require_file(setup_dir / name, f"setup artifact {name}")
        paths[key] = path
    artifacts = manifest.get("artifacts", {})
    digest_keys = {
        "compiled_circuit": "compiled_circuit_sha256",
        "settings": "settings_sha256",
        "pk": "pk_sha256",
        "vk": "vk_sha256",
        "srs": "srs_sha256",
    }
    for key, digest_key in digest_keys.items():
        expected = artifacts.get(digest_key)
        actual = _sha256_file(paths[key])
        if not isinstance(expected, str) or expected != actual:
            raise ZkError(
                f"setup artifact {paths[key].name} does not match the setup manifest; "
                "refusing to reuse a mismatched artifact")
    settings = _read_json(paths["settings"], "settings")
    return manifest, paths, _output_scale(settings), settings


def run_prove(input_path, model_path, setup_dir, credential_path):
    """Generate the witness over private features and a real EZKL proof."""
    if _ezkl_version() != REQUIRED_EZKL_VERSION:
        raise ZkError(f"ezkl {REQUIRED_EZKL_VERSION} is required (found {_ezkl_version()})")
    input_path = _require_file(input_path, "input file")
    document = _read_json(input_path, "input file")
    features = extract_features(document)  # the README's six-feature contract

    model_path = _require_file(model_path, "ONNX model")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    manifest, artifacts, scale, _ = _load_setup(Path(setup_dir))
    if manifest.get("model_sha256") != model_hash:
        raise ZkError("the ONNX model does not match the model used by the setup directory")
    if manifest.get("ezkl_version") != REQUIRED_EZKL_VERSION:
        raise ZkError("the setup directory was produced with a different EZKL version")

    credential_path = Path(credential_path)

    with _workspace("zkml-prove-") as stage:
        data_path = stage / "input.json"
        witness_path = stage / "witness.json"
        proof_path = stage / "proof.pf"
        data_path.write_text(
            json.dumps({"input_data": [features], "output_data": None}), encoding="utf-8")

        _ezkl("generating the witness", ezkl.gen_witness,
              data=str(data_path), model=str(artifacts["compiled_circuit"]),
              output=str(witness_path), vk_path=str(artifacts["vk"]),
              srs_path=str(artifacts["srs"]))
        _ezkl("producing the proof", ezkl.prove,
              witness=str(witness_path), model=str(artifacts["compiled_circuit"]),
              pk_path=str(artifacts["pk"]), proof_path=str(proof_path),
              srs_path=str(artifacts["srs"]))

        # Self-check before issuing a credential: it must verify locally and
        # expose two public scores decoded straight from the proof.
        verified = _ezkl("self-verifying the fresh proof", ezkl.verify,
                         proof_path=str(proof_path), settings_path=str(artifacts["settings"]),
                         vk_path=str(artifacts["vk"]), srs_path=str(artifacts["srs"]))
        if verified is not True:
            raise ZkError("fresh proof did not verify; no credential written")

        proof_bytes = proof_path.read_bytes()
        proof_document = json.loads(proof_bytes.decode("utf-8"))
        if proof_document.get("version") != REQUIRED_EZKL_VERSION:
            raise ZkError("proof was produced with a different EZKL version")
        instances = proof_document.get("instances")
        integers, scores, label = _decode_instances(instances, scale)

        credential = {
            "format_version": CREDENTIAL_FORMAT,
            "ezkl_version": REQUIRED_EZKL_VERSION,
            "model_sha256": model_hash,
            "artifacts": {
                "compiled_circuit_sha256": manifest["artifacts"]["compiled_circuit_sha256"],
                "settings_sha256": manifest["artifacts"]["settings_sha256"],
                "vk_sha256": manifest["artifacts"]["vk_sha256"],
                "srs_sha256": manifest["artifacts"]["srs_sha256"],
            },
            "proof": {
                "format": "ezkl-proof-json",
                "encoding": "base64",
                "data": base64.b64encode(proof_bytes).decode("ascii"),
            },
            "public_output": {
                "output_scale": scale,
                "instances": instances[0],
                "quantized_scores": scores,
                "quantized_integers": integers,
                "label": label,
            },
        }
        serialized = json.dumps(credential, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        _write_bytes_atomic(credential_path, serialized)

    return {"credential": str(credential_path), "model_sha256": model_hash,
            "quantized_scores": scores, "label": label}


def _verify_credential_shape(credential):
    if not isinstance(credential, dict) or set(credential) != CREDENTIAL_KEYS:
        raise ZkError("credential has an unexpected structure")
    if credential["format_version"] != CREDENTIAL_FORMAT:
        raise ZkError(f"unsupported credential format version {credential['format_version']!r}")
    if credential["ezkl_version"] != REQUIRED_EZKL_VERSION:
        raise ZkError("credential was produced with a different EZKL version")
    model_hash = credential["model_sha256"]
    if not isinstance(model_hash, str) or len(model_hash) != 64:
        raise ZkError("credential model_sha256 is invalid")
    artifacts = credential["artifacts"]
    if not isinstance(artifacts, dict):
        raise ZkError("credential artifacts section is invalid")
    for key in ("compiled_circuit_sha256", "settings_sha256", "vk_sha256", "srs_sha256"):
        if not isinstance(artifacts.get(key), str) or len(artifacts[key]) != 64:
            raise ZkError(f"credential artifact digest {key} is invalid")
    proof = credential["proof"]
    if not isinstance(proof, dict) or proof.get("format") != "ezkl-proof-json" \
            or proof.get("encoding") != "base64" or not isinstance(proof.get("data"), str):
        raise ZkError("credential proof section is invalid")
    public_output = credential["public_output"]
    if not isinstance(public_output, dict):
        raise ZkError("credential public_output section is invalid")
    scale_value = public_output.get("output_scale")
    if not isinstance(scale_value, int) or isinstance(scale_value, bool) or scale_value <= 0:
        raise ZkError("credential output_scale is invalid")
    instances = public_output.get("instances")
    if not isinstance(instances, list) or len(instances) != 2 \
            or not all(isinstance(value, str) for value in instances):
        raise ZkError("credential public instances are invalid")
    integers = public_output.get("quantized_integers")
    if not isinstance(integers, list) or len(integers) != 2 \
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in integers):
        raise ZkError("credential quantized_integers are invalid")
    scores = public_output.get("quantized_scores")
    if not isinstance(scores, list) or len(scores) != 2 \
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in scores):
        raise ZkError("credential quantized_scores are invalid")
    if public_output.get("label") not in ("normal", "inspect"):
        raise ZkError("credential label is invalid")


def run_verify(credential_path, model_path, settings_path, vk_path, srs_path):
    """Verify the credential's proof using only verifier-chosen artifacts."""
    if _ezkl_version() != REQUIRED_EZKL_VERSION:
        raise ZkError(f"ezkl {REQUIRED_EZKL_VERSION} is required (found {_ezkl_version()})")

    credential_path = _require_file(credential_path, "credential file")
    credential = _read_json(credential_path, "credential file")
    _verify_credential_shape(credential)

    model_path = _require_file(model_path, "verification model")
    settings_path = _require_file(settings_path, "settings file")
    vk_path = _require_file(vk_path, "verification key")
    srs_path = _require_file(srs_path, "SRS file")

    # Identity checks against the verifier's own files; the credential's
    # embedded hashes never substitute for the verifier's parameters.
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != credential["model_sha256"]:
        raise ZkError("verifier model does not match the model recorded in the credential")
    supplied = {
        "settings_sha256": _sha256_file(settings_path),
        "vk_sha256": _sha256_file(vk_path),
        "srs_sha256": _sha256_file(srs_path),
    }
    for key, actual in supplied.items():
        if actual != credential["artifacts"][key]:
            raise ZkError(f"verifier {key[:-7]} does not match the credential's recorded artifact")

    settings = _read_json(settings_path, "settings file")
    scale = _output_scale(settings)
    if scale != credential["public_output"]["output_scale"]:
        raise ZkError("settings output scale does not match the credential")

    try:
        proof_bytes = base64.b64decode(credential["proof"]["data"], validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ZkError(f"credential proof data is not valid base64: {exc}") from exc
    if not proof_bytes:
        raise ZkError("credential proof data is empty")
    try:
        proof_document = json.loads(proof_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ZkError(f"credential proof is not valid EZKL proof JSON: {exc}") from exc
    if proof_document.get("version") != REQUIRED_EZKL_VERSION:
        raise ZkError("proof was produced with a different EZKL version")

    # Public outputs must come from the proof file itself, never trusted from
    # the credential's summary fields.
    integers, scores, label = _decode_instances(proof_document.get("instances"), scale)
    claimed = credential["public_output"]
    if proof_document["instances"][0] != claimed["instances"]:
        raise ZkError("credential public instances do not match the proof")
    if integers != claimed["quantized_integers"] or label != claimed["label"] \
            or scores != claimed["quantized_scores"]:
        raise ZkError("credential claimed results do not match the proof's public outputs")

    with _workspace("zkml-verify-") as stage:
        proof_path = stage / "proof.pf"
        proof_path.write_bytes(proof_bytes)
        verified = _ezkl("verifying the proof", ezkl.verify,
                         proof_path=str(proof_path), settings_path=str(settings_path),
                         vk_path=str(vk_path), srs_path=str(srs_path))
    if verified is not True:
        raise ZkError("proof verification failed")

    return {
        "verified": True,
        "quantized_scores": scores,
        "label": label,
        "model_sha256": model_hash,
    }
