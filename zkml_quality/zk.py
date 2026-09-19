"""Real EZKL 23.0.5 CPU zero-knowledge proof loop for the quality model.

Three stages:

* ``run_setup`` compiles an ONNX model into a circuit and produces settings,
  proving key, verification key and SRS (offline, CPU), together with a
  ``manifest.json`` that pins every public artifact by SHA-256.
* ``run_prove`` consumes a private six-feature input, generates the witness and
  a real proof, and writes a self-contained JSON credential. Only the two
  quantized output scores are public; the features stay private.
* ``run_verify`` checks a credential against public materials the verifier
  independently obtained: the setup manifest plus the ONNX model, settings,
  VK and SRS named by it. It needs neither the original input nor the proving
  key nor the compiled circuit nor any network access.

Trust model: the verifier's own ``manifest.json`` (obtained through a channel
the verifier trusts, e.g. produced by their own ``zk-setup`` run) is the single
root of trust. It pins the format version, credential/setup kind, EZKL version,
the model digest and the digests of settings, the verification key and the SRS;
the manifest in turn binds the circuit, because the VK only validates proofs
for the circuit/settings it was generated for. Nothing embedded in the
credential is ever treated as authoritative: its digests and parameters are at
most cross-checked against the manifest and, ultimately, against EZKL.

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
import threading
from pathlib import Path

import ezkl

from .inference import validate_features
from .registry import RegistryError, admit_version, require_record_matches

FORMAT_VERSION = 1
CREDENTIAL_KIND = "zk-quality-credential"
SETUP_KIND = "zk-quality-setup"
ARTIFACT_NAMES = {
    "compiled": "compiled.ezkl",
    "settings": "settings.json",
    "pk": "proving.key",
    "vk": "verification.key",
    "srs": "srs",
}
# Public verification materials whose digests the manifest pins and that the
# verifier must hold. The compiled circuit and proving key stay with the prover.
VERIFICATION_ARTIFACTS = ("settings", "vk", "srs")
MANIFEST_NAME = "manifest.json"

# Labels follow the existing inference convention: index 0 is "normal",
# index 1 is "inspect", and a tie resolves to "normal".
LABELS = ("normal", "inspect")

_HEX64 = set("0123456789abcdef")


class ZkError(Exception):
    """Raised for any invalid input, artifact mismatch or EZKL failure."""


class ZkInputError(ZkError):
    """The request or its input document is invalid (bad arguments/features)."""


class ZkArtifactMissing(ZkError):
    """A required artifact (model, setup, credential target directory) is absent."""


class ZkArtifactError(ZkError):
    """An artifact exists but is corrupt or unreadable."""


class ZkDigestMismatch(ZkError):
    """An artifact or credential does not hash to its pinned digest."""


class ZkEzklError(ZkError):
    """EZKL itself failed (witness, proving or verification)."""


class ZkOutputError(ZkError):
    """Writing the output artifact failed."""


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


def _is_sha256_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX64 for char in value)


def _require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise ZkError(f"{description} not found: {path}")
    return path


_QUIET_STDERR_LOCK = threading.Lock()
_QUIET_STDERR_STATE = None  # {"capture": file, "saved": fd, "count": int}


def _release_quiet_stderr():
    """Drop one reference to the shared stderr redirection.

    Only the last leaver restores the original descriptor and drains the
    captured output, so concurrent provers can never strand fd 2 on a closed
    capture file.
    """
    global _QUIET_STDERR_STATE
    with _QUIET_STDERR_LOCK:
        state = _QUIET_STDERR_STATE
        state["count"] -= 1
        if state["count"] > 0:
            return None
        _QUIET_STDERR_STATE = None
        os.dup2(state["saved"], 2)
        os.close(state["saved"])
        capture = state["capture"]
        capture.seek(0)
        tail = capture.read().decode("utf-8", "replace").strip().splitlines()
        capture.close()
        return tail[-1].strip() if tail else None


@contextlib.contextmanager
def _quiet_stderr():
    """Redirect fd-level stderr (EZKL writes via Rust, bypassing Python).

    Captured output is attached to the raised error so EZKL's own diagnostic
    is still surfaced on failure. Concurrent users (e.g. a batch run with
    several workers) share one redirection; the last leaver restores stderr.
    """
    global _QUIET_STDERR_STATE
    with _QUIET_STDERR_LOCK:
        if _QUIET_STDERR_STATE is None:
            captured = tempfile.TemporaryFile(mode="w+b")
            saved = os.dup(2)
            os.dup2(captured.fileno(), 2)
            _QUIET_STDERR_STATE = {"capture": captured, "saved": saved, "count": 1}
        else:
            _QUIET_STDERR_STATE["count"] += 1
    try:
        yield
    except Exception as error:
        detail = _release_quiet_stderr()
        if isinstance(error, ZkError):
            raise
        raise ZkEzklError(f"EZKL failed: {detail if detail is not None else error}") from error
    else:
        _release_quiet_stderr()


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

            # Hash only after the files are final, then validate the document we
            # are about to publish so setup can never emit a broken manifest.
            artifact_digests = {key: _sha256(target) for key, target in targets.items()}
            manifest = {
                "format_version": FORMAT_VERSION,
                "kind": SETUP_KIND,
                "ezkl_version": ezkl_version(),
                "model_sha256": model_sha,
                "logrows": logrows,
                "output_scale": output_scale,
                "artifacts": artifact_digests,
            }
            _validate_manifest_fields(manifest)
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


def _validate_manifest_fields(manifest):
    """Shape and field validation shared by the producer and every consumer.

    Every digest must be 64 lowercase hex characters; missing or malformed
    fields are rejected rather than coerced.
    """
    if not isinstance(manifest, dict):
        raise ZkError("setup manifest must be a JSON object")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ZkError("setup manifest format version is unsupported")
    if manifest.get("kind") != SETUP_KIND:
        raise ZkError("setup manifest is unsupported or was not produced by zk-setup")
    if manifest.get("ezkl_version") != ezkl_version():
        raise ZkError(
            f"setup was produced with EZKL {manifest.get('ezkl_version')}, "
            f"installed EZKL is {ezkl_version()}")
    if not _is_sha256_hex(manifest.get("model_sha256")):
        raise ZkError("setup manifest records an invalid model SHA-256")
    if not isinstance(manifest.get("logrows"), int) or isinstance(manifest.get("logrows"), bool) \
            or not 1 <= manifest["logrows"] <= 30:
        raise ZkError("setup manifest records an invalid logrows value")
    if not isinstance(manifest.get("output_scale"), int) \
            or isinstance(manifest.get("output_scale"), bool) or manifest["output_scale"] <= 0:
        raise ZkError("setup manifest records an invalid output scale")
    digests = manifest.get("artifacts")
    if not isinstance(digests, dict):
        raise ZkError("setup manifest does not record artifact digests")
    for key in ARTIFACT_NAMES:
        if not _is_sha256_hex(digests.get(key)):
            raise ZkError(f"setup manifest records an invalid digest for '{ARTIFACT_NAMES[key]}'")
    if set(digests) != set(ARTIFACT_NAMES):
        raise ZkError("setup manifest records an unexpected set of artifacts")


def _read_manifest(manifest_path):
    """Load and strictly validate the verifier's own setup manifest."""
    manifest_path = _require_file(manifest_path, "setup manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"),
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValueError(f"invalid JSON constant {value}")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ZkError(f"invalid setup manifest JSON: {error}") from error
    _validate_manifest_fields(manifest)
    return manifest


def load_verifier_materials(manifest_path, model, settings_path, vk_path, srs_path):
    """Resolve the verifier's trust material against its own manifest.

    The manifest is the root of trust: the ONNX model and each verification
    artifact must hash to the digest pinned in it. Returns the validated
    manifest, material paths and the output scale. Nothing here consults a
    credential.
    """
    manifest = _read_manifest(manifest_path)
    supplied = {
        "settings": (settings_path, "settings file"),
        "vk": (vk_path, "verification key"),
        "srs": (srs_path, "SRS file"),
    }
    materials = {
        "model": (_require_file(model, "ONNX model"), manifest["model_sha256"]),
    }
    for key in VERIFICATION_ARTIFACTS:
        path, description = supplied[key]
        materials[key] = (
            _require_file(path, description), manifest["artifacts"][key])
    for key, (path, expected_digest) in materials.items():
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            label = "ONNX model" if key == "model" else ARTIFACT_NAMES[key]
            raise ZkError(
                f"{label} does not match the verifier manifest: {path}. "
                "Use only the public materials pinned by a zk-setup manifest you trust")
    return {
        "manifest": manifest,
        "paths": {key: path for key, (path, _) in materials.items()},
        "model_sha256": manifest["model_sha256"],
        "output_scale": int(manifest["output_scale"]),
    }


def _artifact_paths(setup_dir):
    return {key: _require_file(setup_dir / name, f"setup artifact '{name}'")
            for key, name in ARTIFACT_NAMES.items()}


def _check_artifacts(manifest, paths):
    """Verify every setup artifact on disk still matches the setup manifest."""
    digests = manifest["artifacts"]
    for key in paths:
        if _sha256(paths[key]) != digests[key]:
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


def _require_input_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise ZkArtifactMissing(f"{description} not found")
    return path


def _load_prove_features(input_path):
    """Read and validate the private feature vector, without revealing it."""
    try:
        document = json.loads(input_path.read_text(encoding="utf-8"),
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValueError(f"invalid JSON constant {value}")))
    except json.JSONDecodeError as error:
        raise ZkInputError(f"input file is not valid JSON: {error.msg}") from error
    except OSError as error:
        raise ZkArtifactError(f"input file cannot be read: {error.strerror}") from error
    except ValueError as error:
        raise ZkInputError(str(error)) from error
    try:
        return validate_features(document)  # identical six-dimension contract as `infer`
    except ValueError as error:
        raise ZkInputError(str(error)) from error


def _prepare_prove(input_path, model, setup_dir, credential_path):
    """Validate every private/public input of a proof request, without proving.

    Returns only non-sensitive, reusable facts: the validated feature vector
    (kept in memory by the caller, never persisted), the setup paths, the
    model digest and the output scale.
    """
    input_path = _require_input_file(input_path, "input file")
    model = _require_input_file(model, "ONNX model")
    setup_dir = Path(setup_dir)
    if not setup_dir.is_dir():
        raise ZkArtifactMissing("setup directory not found")
    credential_path = Path(credential_path)
    if credential_path.is_dir():
        raise ZkInputError("credential output path is a directory")
    if not credential_path.resolve().parent.is_dir():
        raise ZkArtifactMissing("credential output directory does not exist")

    features = _load_prove_features(input_path)

    try:
        manifest = _read_manifest(setup_dir / MANIFEST_NAME)
    except ZkArtifactMissing:
        raise
    except ZkError as error:
        raise ZkArtifactError(str(error)) from error
    try:
        model_sha = _sha256(model)
    except OSError as error:
        raise ZkArtifactError(f"ONNX model cannot be read: {error.strerror or error}") from error
    if model_sha != manifest["model_sha256"]:
        raise ZkDigestMismatch("model does not match the model this setup was built for")
    try:
        paths = _artifact_paths(setup_dir)
        _check_artifacts(manifest, paths)
    except ZkArtifactMissing:
        raise
    except ZkError as error:
        raise ZkDigestMismatch(str(error)) from error
    scale = int(manifest["output_scale"])
    return features, paths, model_sha, scale, credential_path


def _issue_credential(features, paths, model_sha, scale, credential_path):
    """Run the real EZKL witness/prove/self-verify and publish a credential."""
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
                raise ZkEzklError("witness generation produced no public outputs")
            proof_result = ezkl.prove(str(witness_path), str(paths["compiled"]),
                                      str(paths["pk"]), str(proof_path), str(paths["srs"]))
            if not proof_path.is_file() or not isinstance(proof_result, dict):
                raise ZkEzklError("proving did not produce a proof")
            # Self-check before issuing a credential; requires local artifacts only.
            if ezkl.verify(str(proof_path), str(paths["settings"]), str(paths["vk"]),
                           str(paths["srs"]), False) is not True:
                raise ZkEzklError("freshly generated proof failed local verification")

        proof_doc = _read_proof_document(proof_path)
        scores, fixed_point, label = _decode_instances(proof_doc["instances"], scale)

        # The witness also reports the decoded outputs; they must agree with the
        # public instances that actually went into the proof.
        witness_scores, _, witness_label = _decode_instances(witness["outputs"], scale)
        if witness_scores != scores or witness_label != label:
            raise ZkEzklError("witness outputs and proof instances disagree")

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
    try:
        _write_credential(credential, credential_path)
    except OSError as error:
        raise ZkOutputError(f"cannot write credential: {error.strerror or error}") from error
    return scores, label


def run_prove(input_path, model, setup_dir, credential_path):
    """Generate witness and a real proof for one private feature vector."""
    features, paths, model_sha, scale, credential_path = _prepare_prove(
        input_path, model, setup_dir, credential_path)
    scores, label = _issue_credential(
        features, paths, model_sha, scale, credential_path)
    credential_sha = _sha256(credential_path)
    return {"credential": str(credential_path), "model_sha256": model_sha,
            "credential_sha256": credential_sha,
            "quantized_scores": scores, "label": label}


def _write_credential(credential, credential_path):
    credential_path = Path(credential_path)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=credential_path.resolve().parent,
        prefix=".credential-", suffix=".tmp", delete=False)
    try:
        json.dump(credential, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
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
    if not _is_sha256_hex(credential["model_sha256"]):
        raise ZkError("credential model digest is invalid")
    # A credential must never carry the private input.
    if "features" in credential or "input_data" in credential:
        raise ZkError("credential illegally contains private input data")
    artifacts = credential["verification_artifacts"]
    if not isinstance(artifacts, dict) or any(
            not _is_sha256_hex(artifacts.get(key))
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


def run_verify(credential_path, manifest_path, model, settings_path, vk_path, srs_path,
               registry_path, model_version):
    """Verify a credential offline against the verifier's own trust material.

    Before any cryptographic check the verifier's local model registry must
    admit ``model_version`` (registered and ``enabled``) and the admitted
    record must agree, field by field, with the verifier's own trusted
    manifest: model and manifest SHA-256, EZKL version, output scale and the
    settings/VK/SRS digests. Unknown, disabled, revoked or mismatching
    versions, and a corrupt registry, are all refused.

    The verifier's ``manifest.json`` — not the credential and not the registry
    record — stays the root of trust. The model digest and every
    settings/VK/SRS digest are recomputed from the verifier's files and matched
    to the manifest first; the matching files then drive EZKL cryptographic
    verification. Credential-embedded digests and parameters are never
    authoritative: they are merely required to agree with the manifest, and a
    proof for a different circuit/key cannot pass EZKL against the pinned VK
    and settings.
    """
    # Registry gate first: before hashing anything or touching the credential,
    # only an explicitly enabled version in a valid verifier-owned registry
    # may be verified at all. A missing/corrupt registry or an unknown,
    # disabled or revoked version stops verification here.
    record = admit_version(registry_path, model_version)

    # Bind model, circuit (via the VK generated for it) and verification
    # parameters to the verifier's manifest, then compare the admitted record
    # against it field by field. Both consult only verifier-owned files;
    # nothing in the credential can affect either decision.
    materials = load_verifier_materials(
        manifest_path, model, settings_path, vk_path, srs_path)
    manifest = materials["manifest"]
    paths = materials["paths"]
    model_sha = materials["model_sha256"]
    scale = materials["output_scale"]
    require_record_matches(record, manifest_sha=_sha256(manifest_path), manifest=manifest)

    credential = _load_credential(credential_path)

    # Cross-checks against the manifest. These are defence in depth — the
    # credential values cannot weaken verification because EZKL is driven
    # solely by the manifest-pinned files — but a credential that claims a
    # different model or parameters is rejected outright rather than accepted
    # over a valid proof.
    if credential["model_sha256"] != model_sha:
        raise ZkError("credential model digest does not match the verifier manifest")
    claimed = credential["verification_artifacts"]
    expected_artifacts = {
        "settings_sha256": manifest["artifacts"]["settings"],
        "vk_sha256": manifest["artifacts"]["vk"],
        "srs_sha256": manifest["artifacts"]["srs"],
    }
    for key, expected_digest in expected_artifacts.items():
        if claimed[key] != expected_digest:
            raise ZkError(f"credential {key} does not match the verifier manifest")
    if credential["ezkl_version"] != manifest["ezkl_version"]:
        raise ZkError("credential EZKL version does not match the verifier manifest")
    if credential["public_output"]["output_scale"] != scale:
        raise ZkError("credential output scale does not match the verifier manifest")

    # Independently confirm the pinned settings file really carries the scale
    # and EZKL version the manifest/credential describe.
    try:
        settings_doc = json.loads(paths["settings"].read_text(encoding="utf-8"))
        settings_scale = int(settings_doc["model_output_scales"][0])
        settings_version = str(settings_doc["version"])
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise ZkError(f"settings file is unreadable: {error}") from error
    if settings_version != manifest["ezkl_version"]:
        raise ZkError(
            f"settings were produced with EZKL {settings_version}, "
            f"manifest pins EZKL {manifest['ezkl_version']}")
    if settings_scale != scale:
        raise ZkError("settings output scale does not match the verifier manifest")

    with tempfile.TemporaryDirectory(prefix="zkverify-") as scratch:
        proof_path = Path(scratch) / "proof.pf"
        proof_path.write_text(json.dumps(credential["proof"]), encoding="utf-8")
        with _quiet_stderr():
            accepted = ezkl.verify(str(proof_path), str(paths["settings"]), str(paths["vk"]),
                                   str(paths["srs"]), False)
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
