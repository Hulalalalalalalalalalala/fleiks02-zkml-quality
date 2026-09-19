"""Local offline model registry for the zk-quality verifier.

The registry is a single JSON document recording, per model version, the
digests of the exact manifest, ONNX model, settings, verification key and SRS
the verifier has reviewed and approved, together with a lifecycle status:

* ``disabled`` -- freshly registered; not yet admitted for verification.
* ``enabled`` -- admitted: ``zk-verify`` accepts credentials for this version.
* ``revoked`` -- permanently retired; this state is irreversible.

Versions match ``[A-Za-z0-9._-]+``. Registering the same version with the same
content is an idempotent no-op; a conflicting re-registration is refused
rather than overwriting. Every update is atomic (temporary file in the same
directory, then ``os.replace``), a missing registry is created on first
register, and a corrupt or structurally invalid registry is rejected without
modifying the file. No network access is ever needed.
"""
import contextlib
import json
import os
import re
import tempfile
from pathlib import Path

from .zk import ZkError, _is_sha256_hex, _read_manifest, _require_file, _sha256

REGISTRY_FORMAT_VERSION = 1
REGISTRY_KIND = "zk-quality-model-registry"
STATUSES = ("disabled", "enabled", "revoked")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
RECORD_FIELDS = ("status", "model_sha256", "manifest_sha256", "ezkl_version",
                 "output_scale", "artifacts")
RECORD_ARTIFACTS = ("settings", "vk", "srs")


def require_version(version):
    """Validate a model version string against the registry contract."""
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ZkError(f"invalid model version {version!r}: must match [A-Za-z0-9._-]+")
    return version


def _validate_record(record, version):
    if not isinstance(record, dict) or set(record) != set(RECORD_FIELDS):
        raise ZkError(f"record for '{version}' is malformed")
    if record["status"] not in STATUSES:
        raise ZkError(f"record for '{version}' has an invalid status")
    for key in ("model_sha256", "manifest_sha256"):
        if not _is_sha256_hex(record[key]):
            raise ZkError(f"record for '{version}' records an invalid {key}")
    if not isinstance(record["ezkl_version"], str) or not record["ezkl_version"]:
        raise ZkError(f"record for '{version}' records an invalid EZKL version")
    scale = record["output_scale"]
    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ZkError(f"record for '{version}' records an invalid output scale")
    artifacts = record["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(RECORD_ARTIFACTS) \
            or any(not _is_sha256_hex(artifacts[key]) for key in RECORD_ARTIFACTS):
        raise ZkError(f"record for '{version}' records invalid artifact digests")


def _validate_registry(document):
    """Strict structural validation; anything unexpected is rejected."""
    if not isinstance(document, dict) \
            or set(document) != {"format_version", "kind", "models"}:
        raise ZkError("registry must be a JSON object with format_version, kind and models")
    if document["format_version"] != REGISTRY_FORMAT_VERSION \
            or document["kind"] != REGISTRY_KIND:
        raise ZkError("registry has an unsupported format version or kind")
    models = document["models"]
    if not isinstance(models, dict):
        raise ZkError("registry models must be a JSON object")
    for version, record in models.items():
        require_version(version)
        _validate_record(record, version)
    return document


def _empty_registry():
    return {"format_version": REGISTRY_FORMAT_VERSION,
            "kind": REGISTRY_KIND, "models": {}}


def _load_registry(registry_path, allow_missing=False):
    """Load and strictly validate the registry; never modifies the file."""
    registry_path = Path(registry_path)
    if not registry_path.exists():
        if allow_missing:
            return _empty_registry()
        raise ZkError(f"model registry not found: {registry_path}")
    if not registry_path.is_file():
        raise ZkError(f"model registry is not a regular file: {registry_path}")
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"),
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValueError(f"invalid JSON constant {value}")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ZkError(f"model registry is corrupt or unreadable: {error}") from error
    try:
        _validate_registry(document)
    except ZkError as error:
        raise ZkError(f"model registry is invalid: {error}") from error
    return document


def _write_registry(registry_path, document):
    """Atomically replace the registry; a failure leaves the old file intact."""
    _validate_registry(document)  # never publish a broken registry
    registry_path = Path(registry_path)
    parent = registry_path.resolve().parent
    if not parent.is_dir():
        raise ZkError(f"registry directory does not exist: {parent}")
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=parent,
        prefix=".registry-", suffix=".tmp", delete=False)
    try:
        json.dump(document, handle, ensure_ascii=False, allow_nan=False,
                  indent=2, sort_keys=True)
        handle.write("\n")
        handle.close()
        os.replace(handle.name, registry_path)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def register_model(registry_path, version, manifest_path, model_path):
    """Register a caller-endorsed manifest and ONNX model under ``version``.

    The manifest is fully validated and the model must hash to the digest the
    manifest pins. The new record starts out ``disabled``. Re-registering the
    same version with identical content is an idempotent no-op; different
    content is refused, and a revoked record can never be overwritten.
    """
    version = require_version(version)
    manifest_path = _require_file(manifest_path, "setup manifest")
    model_path = _require_file(model_path, "ONNX model")
    manifest = _read_manifest(manifest_path)
    model_sha = _sha256(model_path)
    if model_sha != manifest["model_sha256"]:
        raise ZkError("ONNX model does not match the manifest being registered")
    record = {
        "status": "disabled",
        "model_sha256": model_sha,
        "manifest_sha256": _sha256(manifest_path),
        "ezkl_version": manifest["ezkl_version"],
        "output_scale": manifest["output_scale"],
        "artifacts": {key: manifest["artifacts"][key] for key in RECORD_ARTIFACTS},
    }
    registry = _load_registry(registry_path, allow_missing=True)
    existing = registry["models"].get(version)
    if existing is not None:
        if any(existing[key] != record[key] for key in RECORD_FIELDS if key != "status"):
            raise ZkError(
                f"model version '{version}' is already registered with different "
                "content; refusing to overwrite")
        return {"version": version, "status": existing["status"], "changed": False}
    registry["models"][version] = record
    _write_registry(registry_path, registry)
    return {"version": version, "status": "disabled", "changed": True}


def list_models(registry_path):
    """Return all records as dicts, ordered lexicographically by version."""
    registry = _load_registry(registry_path, allow_missing=True)
    return [{"version": version, **registry["models"][version]}
            for version in sorted(registry["models"])]


def _transition(registry_path, version, target):
    version = require_version(version)
    registry = _load_registry(registry_path)
    record = registry["models"].get(version)
    if record is None:
        raise ZkError(f"model version '{version}' is not registered")
    status = record["status"]
    if status == "revoked" and target != "revoked":
        raise ZkError(f"model version '{version}' is revoked and cannot be re-enabled")
    if status != target:
        record["status"] = target
        _write_registry(registry_path, registry)
    return {"version": version, "status": target}


def enable_model(registry_path, version):
    """Move a disabled record to enabled; repeated enables are idempotent."""
    return _transition(registry_path, version, "enabled")


def revoke_model(registry_path, version):
    """Irreversibly revoke a record; repeated revokes are idempotent."""
    return _transition(registry_path, version, "revoked")


def require_enabled_record(registry_path, version):
    """Admission gate for ``zk-verify``: only enabled records pass.

    A missing or corrupt registry, an unregistered version, or a record that
    is not enabled are all rejected.
    """
    version = require_version(version)
    registry = _load_registry(registry_path)
    record = registry["models"].get(version)
    if record is None:
        raise ZkError(f"model version '{version}' is not registered")
    if record["status"] != "enabled":
        raise ZkError(
            f"model version '{version}' is {record['status']}; "
            "only enabled records are admitted for verification")
    return record
