"""Local, fully offline model registry for zk-quality models.

The registry is a single verifier-owned JSON file mapping caller-chosen model
versions to the pinned facts of a ``zk-setup`` run the caller approves:

* SHA-256 of the ONNX model and of the setup ``manifest.json``;
* the exact EZKL version and circuit ``output_scale``;
* SHA-256 digests of the settings, verification key and SRS.

A newly registered version starts ``disabled`` and must be explicitly
``enable``d before ``zk-verify`` will admit it. ``revoke`` is irreversible: a
revoked version can never be enabled or overwritten. Registering the same
version with the same content is a no-op (idempotent); any content conflict is
rejected and never overwrites the existing record.

The registry file is treated as hostile input on every load: JSON syntax,
duplicate keys, the envelope, every version key and every record field are
strictly validated. A missing, corrupt or structurally illegal registry is
rejected and is never rewritten, so a failed update always leaves the previous
file byte-for-byte intact. All successful updates are atomic (temp file plus
``os.replace``).
"""
import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

FORMAT_VERSION = 1
REGISTRY_KIND = "model-registry"
VERSION_RE = re.compile(r"[A-Za-z0-9._-]+")
EZKL_VERSION_RE = re.compile(r"[A-Za-z0-9._+-]+")
STATUS_DISABLED = "disabled"
STATUS_ENABLED = "enabled"
STATUS_REVOKED = "revoked"
STATUSES = (STATUS_DISABLED, STATUS_ENABLED, STATUS_REVOKED)
# Verification parameters pinned per version; the compiled circuit and proving
# key never enter the registry because verifiers never hold them.
PINNED_ARTIFACTS = ("settings", "vk", "srs")

_HEX64 = set("0123456789abcdef")


class RegistryError(Exception):
    """Raised for any missing, corrupt, conflicting or illegal registry use."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX64 for char in value)


def validate_version(version):
    """A version is a non-empty ``[A-Za-z0-9._-]+`` token; nothing else."""
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise RegistryError(
            "model version must be a non-empty token matching [A-Za-z0-9._-]+")
    return version


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise RegistryError(f"registry JSON contains a duplicate key: {key!r}")
        document[key] = value
    return document


def _validate_document(document):
    if not isinstance(document, dict):
        raise RegistryError("registry must be a JSON object")
    if set(document) != {"format_version", "kind", "models"}:
        raise RegistryError("registry has missing or unexpected top-level fields")
    if document["format_version"] != FORMAT_VERSION:
        raise RegistryError("registry format version is unsupported")
    if document["kind"] != REGISTRY_KIND:
        raise RegistryError("registry file is not a model registry")
    models = document["models"]
    if not isinstance(models, dict):
        raise RegistryError("registry 'models' must be a JSON object")

    for version, record in models.items():
        validate_version(version)
        if not isinstance(record, dict):
            raise RegistryError(f"registry record for {version!r} must be an object")
        if set(record) != {"status", "model_sha256", "manifest_sha256", "ezkl_version",
                           "output_scale", "artifacts"}:
            raise RegistryError(f"registry record for {version!r} has missing or extra fields")
        if record["status"] not in STATUSES:
            raise RegistryError(f"registry record for {version!r} has an invalid status")
        if not _is_sha256_hex(record["model_sha256"]):
            raise RegistryError(f"registry record for {version!r} has an invalid model digest")
        if not _is_sha256_hex(record["manifest_sha256"]):
            raise RegistryError(f"registry record for {version!r} has an invalid manifest digest")
        ezkl_version = record["ezkl_version"]
        if not isinstance(ezkl_version, str) or not EZKL_VERSION_RE.fullmatch(ezkl_version):
            raise RegistryError(f"registry record for {version!r} has an invalid EZKL version")
        scale = record["output_scale"]
        if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
            raise RegistryError(f"registry record for {version!r} has an invalid output scale")
        artifacts = record["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(PINNED_ARTIFACTS):
            raise RegistryError(f"registry record for {version!r} has invalid artifact digests")
        if any(not _is_sha256_hex(artifacts[key]) for key in PINNED_ARTIFACTS):
            raise RegistryError(f"registry record for {version!r} has an invalid artifact digest")


def _empty_document():
    return {"format_version": FORMAT_VERSION, "kind": REGISTRY_KIND, "models": {}}


def load_registry(registry_path):
    """Load and strictly validate a registry that must already exist."""
    registry_path = Path(registry_path)
    if registry_path.is_dir():
        raise RegistryError(f"registry path is a directory, not a file: {registry_path}")
    if not registry_path.is_file():
        raise RegistryError(f"model registry not found: {registry_path}")
    try:
        raw = registry_path.read_text(encoding="utf-8")
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  RegistryError(f"invalid JSON constant {value}")))
    except OSError as error:
        raise RegistryError(f"cannot read model registry: {error}") from error
    except json.JSONDecodeError as error:
        raise RegistryError(f"model registry is not valid JSON: {error}") from error
    _validate_document(document)
    return document


def _atomic_write(registry_path, document):
    """Validate, then publish the document via a same-directory temp file."""
    _validate_document(document)
    registry_path = Path(registry_path)
    if registry_path.is_dir():
        raise RegistryError(f"registry path is a directory: {registry_path}")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=registry_path.parent,
        prefix=".registry-", suffix=".tmp", delete=False)
    try:
        json.dump(document, handle, ensure_ascii=False, allow_nan=False,
                  indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, registry_path)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def _record_from_manifest(manifest, manifest_sha):
    return {
        "status": STATUS_DISABLED,
        "model_sha256": manifest["model_sha256"],
        "manifest_sha256": manifest_sha,
        "ezkl_version": manifest["ezkl_version"],
        "output_scale": int(manifest["output_scale"]),
        "artifacts": {key: manifest["artifacts"][key] for key in PINNED_ARTIFACTS},
    }


def register_model(registry_path, version, manifest_path, model_path):
    """Register an approved manifest/ONNX pair under ``version``.

    The setup manifest is fully re-validated (including the installed EZKL
    version) and the ONNX file must hash to the model digest it pins. A new
    version is stored as ``disabled``. Re-registering the identical record is
    a no-op and leaves the current status untouched; any differing content for
    an existing version is a conflict and is rejected without writing.
    """
    version = validate_version(version)
    manifest_path = Path(manifest_path)
    model_path = Path(model_path)
    if not manifest_path.is_file():
        raise RegistryError(f"setup manifest not found: {manifest_path}")
    if not model_path.is_file():
        raise RegistryError(f"ONNX model not found: {model_path}")

    # Imported lazily: zk.py imports this module, so a module-level import
    # would be circular.
    from .zk import ZkError, _read_manifest

    try:
        manifest = _read_manifest(manifest_path)
    except ZkError as error:
        raise RegistryError(f"cannot register an untrusted manifest: {error}") from error
    model_sha = _sha256(model_path)
    if model_sha != manifest["model_sha256"]:
        raise RegistryError(
            "ONNX model does not match the model digest pinned in the manifest; "
            "register only a model/manifest pair you approve")
    manifest_sha = _sha256(manifest_path)
    candidate = _record_from_manifest(manifest, manifest_sha)

    registry_path = Path(registry_path)
    if registry_path.exists():
        document = load_registry(registry_path)
        existing = document["models"].get(version)
        if existing is not None:
            # Identity is content, not status: the lifecycle state must never
            # be resettable by re-registering (a revoked version stays revoked).
            if {key: value for key, value in existing.items() if key != "status"} != \
                    {key: value for key, value in candidate.items() if key != "status"}:
                raise RegistryError(
                    f"model version {version!r} already exists with different content; "
                    "refusing to overwrite. Revoke it or register under a new version")
            return {"version": version, "status": existing["status"], "changed": False}
    else:
        document = _empty_document()

    document["models"][version] = candidate
    _atomic_write(registry_path, document)
    return {"version": version, "status": STATUS_DISABLED, "changed": True}


def list_models(registry_path):
    """Return all records ordered by version in lexicographic order."""
    document = load_registry(registry_path)
    return [dict(version=version, **document["models"][version])
            for version in sorted(document["models"])]


def _set_status(registry_path, version, target, *, allowed_from, idempotent):
    version = validate_version(version)
    document = load_registry(registry_path)
    record = document["models"].get(version)
    if record is None:
        raise RegistryError(f"model version is not registered: {version!r}")
    if record["status"] == target and idempotent:
        return {"version": version, "status": target, "changed": False}
    if record["status"] not in allowed_from:
        raise RegistryError(
            f"model version {version!r} is {record['status']} and cannot be {target}")
    record["status"] = target
    _atomic_write(registry_path, document)
    return {"version": version, "status": target, "changed": True}


def enable_model(registry_path, version):
    """Enable a ``disabled`` version. Enabled stays enabled; revoked is final."""
    return _set_status(registry_path, version, STATUS_ENABLED,
                       allowed_from=(STATUS_DISABLED,), idempotent=True)


def revoke_model(registry_path, version):
    """Irreversibly revoke a registered version; revoking again is a no-op."""
    return _set_status(registry_path, version, STATUS_REVOKED,
                       allowed_from=(STATUS_DISABLED, STATUS_ENABLED), idempotent=True)


def admit_version(registry_path, version):
    """Return the record for ``version`` only if it is registered and enabled.

    This is the registry gate that runs before any cryptographic verification:
    a corrupt registry, an unknown version, or a disabled/revoked version are
    all refused and nothing about a credential can change that decision.
    """
    version = validate_version(version)
    document = load_registry(registry_path)
    record = document["models"].get(version)
    if record is None:
        raise RegistryError(f"model version is not registered: {version!r}")
    if record["status"] != STATUS_ENABLED:
        raise RegistryError(
            f"model version {version!r} is {record['status']}; "
            f"only {STATUS_ENABLED!r} versions can be verified")
    return record


def require_record_matches(record, *, manifest_sha, manifest):
    """Bind an admitted registry record to the verifier's trusted manifest.

    The registry is verifier-owned but is still compared field by field
    against the manifest root of trust: model and manifest digest, EZKL
    version, output scale, and settings/VK/SRS digests must all agree. The
    on-disk files were already hashed against the manifest by
    ``load_verifier_materials``, so a mismatched record — like a forged
    credential — can never steer verification onto other material.
    """
    if record["manifest_sha256"] != manifest_sha:
        raise RegistryError(
            "registry record does not match the verifier manifest: manifest digest differs")
    comparisons = (
        ("model_sha256", record["model_sha256"], manifest["model_sha256"], "model"),
        ("ezkl_version", record["ezkl_version"], manifest["ezkl_version"], "EZKL version"),
        ("output_scale", int(record["output_scale"]), int(manifest["output_scale"]),
         "output scale"),
    )
    for field, actual, expected, label in comparisons:
        if actual != expected:
            raise RegistryError(
                f"registry record does not match the verifier manifest: {label} differs")
    for key in PINNED_ARTIFACTS:
        if record["artifacts"][key] != manifest["artifacts"][key]:
            raise RegistryError(
                f"registry record does not match the verifier manifest: {key} digest differs")
