"""Offline batch verification of ``zk-quality-credential`` documents.

``zk-verify-batch`` is the batch form of ``zk-verify``. It reuses the exact
same seven verifier-chosen trust parameters (registry, model version,
manifest, model, settings, VK, SRS) and the exact same trust chain; only the
credentials differ, arriving as a list in a ``--file`` batch descriptor
instead of one ``--credential`` argument.

Two phases, strictly ordered:

1. The batch descriptor is parsed and validated *before anything is checked*:
   the top level must be an object containing only a non-empty ``items`` array
   of at most :data:`MAX_ITEMS`; every item must be an object containing only
   a unique ``id`` token (``[A-Za-z0-9._-]+``) and a ``credential`` path.
   Duplicate JSON keys, unknown fields, a bad id, a repeated id or any other
   illegal structure rejects the **whole batch** before verification starts.
2. The verifier trust material is resolved **once** (registry admission,
   manifest-pinned model/settings/VK/SRS, registry/manifest agreement and the
   settings cross-check). Any failure aborts the whole batch: no item is
   verified and no per-item result exists.

Only after both gates pass are the items processed in input order, each with
the real EZKL verifier. A single bad credential never blocks the others: a
missing/unreadable credential is ``credential_missing``; malformed JSON,
structure or embedded digests/parameters are ``credential_invalid``; a
tampered proof or public output, or an EZKL rejection, is
``verification_failed``. The command exits 0 once processing finishes even
when some (or all) items were rejected.

Every response — success envelope or fatal error — is a single safe JSON
object. Fatal errors go to stderr with a fixed code and message and empty
stdout; per-item rejections carry only ``id``, ``accepted: false`` and the
fixed ``error`` block. Nothing ever leaks a path, feature, proof byte or
raw exception text, and nothing is written to disk.
"""
import json
import re
from pathlib import Path

from .registry import RegistryError
from .zk import (
    ZkError,
    _check_credential_claims,
    _decode_instances,
    _load_credential,
    _run_ezkl_verify,
    prepare_verifier_context,
)

MAX_ITEMS = 256
_ID_RE = re.compile(r"[A-Za-z0-9._-]+")

# Fatal, whole-batch error codes (nonzero exit, one safe JSON on stderr).
CODE_INVALID_BATCH = "invalid_batch"
CODE_TRUST_CHECK_FAILED = "trust_check_failed"
CODE_INTERNAL_ERROR = "internal_error"

# Per-item rejection codes (exit 0, item rejected, other items continue).
CODE_CREDENTIAL_MISSING = "credential_missing"
CODE_CREDENTIAL_INVALID = "credential_invalid"
CODE_VERIFICATION_FAILED = "verification_failed"

# Fixed, non-revealing messages. A message never embeds a path, a digest,
# credential content or exception text, so no failure can leak input material.
SAFE_MESSAGES = {
    CODE_INVALID_BATCH: "the batch descriptor is invalid",
    CODE_TRUST_CHECK_FAILED: "the verifier trust check failed",
    CODE_INTERNAL_ERROR: "the batch verification command failed",
    CODE_CREDENTIAL_MISSING: "the credential is missing",
    CODE_CREDENTIAL_INVALID: "the credential is invalid",
    CODE_VERIFICATION_FAILED: "credential verification failed",
}


class BatchVerifyError(Exception):
    """A fatal batch error: malformed descriptor or failed trust preflight."""

    def __init__(self, code):
        super().__init__(SAFE_MESSAGES[code])
        self.code = code


class _DuplicateKey(ValueError):
    """Raised by the JSON parser hook when an object repeats a key."""


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateKey(f"duplicate JSON key: {key!r}")
        document[key] = value
    return document


def load_batch_descriptor(file_path):
    """Strictly parse and validate the batch descriptor.

    The descriptor is treated as hostile input. On success returns the items
    in input order as ``(id, credential)`` pairs. Any malformed document —
    unreadable/non-JSON file, duplicate keys, a top-level shape other than
    ``{"items": [...]}``, an empty/too-large/non-array ``items``, an item that
    is not exactly ``{"id", "credential"}``, an illegal or repeated id, or a
    non-string credential — raises :class:`BatchVerifyError` with
    :data:`CODE_INVALID_BATCH`; nothing about the cause is surfaced.
    """
    try:
        raw = Path(file_path).read_text(encoding="utf-8")
    except OSError:
        raise BatchVerifyError(CODE_INVALID_BATCH) from None
    try:
        document = json.loads(
            raw, object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                _DuplicateKey(f"invalid JSON constant {value}")))
    except (json.JSONDecodeError, ValueError):
        raise BatchVerifyError(CODE_INVALID_BATCH) from None

    if not isinstance(document, dict) or set(document) != {"items"}:
        raise BatchVerifyError(CODE_INVALID_BATCH)
    items = document["items"]
    if not isinstance(items, list) or not items or len(items) > MAX_ITEMS:
        raise BatchVerifyError(CODE_INVALID_BATCH)

    parsed = []
    seen_ids = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "credential"}:
            raise BatchVerifyError(CODE_INVALID_BATCH)
        ident = item["id"]
        credential = item["credential"]
        if not isinstance(ident, str) or not _ID_RE.fullmatch(ident):
            raise BatchVerifyError(CODE_INVALID_BATCH)
        if not isinstance(credential, str) or not credential:
            raise BatchVerifyError(CODE_INVALID_BATCH)
        if ident in seen_ids:
            raise BatchVerifyError(CODE_INVALID_BATCH)
        seen_ids.add(ident)
        parsed.append((ident, credential))
    return parsed


def _rejected(ident, code):
    return {"id": ident, "accepted": False,
            "error": {"code": code, "message": SAFE_MESSAGES[code]}}


def _verify_one(ident, credential_path, context):
    """Verify one item, never raising and never leaking failure detail.

    Missing/unreadable credential -> ``credential_missing``; malformed
    credential structure/digests/parameters -> ``credential_invalid``;
    cryptographic failure or tampered proof/public output ->
    ``verification_failed``.
    """
    path = Path(credential_path)
    try:
        if not path.is_file():
            return _rejected(ident, CODE_CREDENTIAL_MISSING)
    except OSError:
        return _rejected(ident, CODE_CREDENTIAL_MISSING)

    try:
        credential = _load_credential(path)
    except ZkError:
        return _rejected(ident, CODE_CREDENTIAL_INVALID)
    except OSError:
        return _rejected(ident, CODE_CREDENTIAL_MISSING)

    # Embedded model/parameter digests and versions that disagree with the
    # verifier manifest are credential digest errors, not proof failures.
    try:
        _check_credential_claims(credential, context)
    except ZkError:
        return _rejected(ident, CODE_CREDENTIAL_INVALID)

    try:
        if not _run_ezkl_verify(credential, context):
            return _rejected(ident, CODE_VERIFICATION_FAILED)
        scores, fixed_point, label = _decode_instances(
            credential["proof"]["instances"], context["settings_scale"])
        if scores != credential["public_output"]["quantized_scores"] \
                or label != credential["public_output"]["label"]:
            return _rejected(ident, CODE_VERIFICATION_FAILED)
    except ZkError:
        return _rejected(ident, CODE_VERIFICATION_FAILED)
    except Exception:  # noqa: BLE001 - per-item safety net, message is fixed
        return _rejected(ident, CODE_VERIFICATION_FAILED)

    return {"id": ident, "accepted": True, "verified": True,
            "model_sha256": context["model_sha256"],
            "quantized_scores": scores,
            "scores_fixed_point": fixed_point, "label": label}


def run_batch_verify(file_path, registry_path, model_version, manifest_path,
                     model, settings_path, vk_path, srs_path):
    """Validate the descriptor, resolve trust once, then verify every item.

    A malformed descriptor or any failure of the one-time trust preflight
    raises :class:`BatchVerifyError` and aborts the whole batch. Otherwise the
    items are verified in input order and the summary plus per-item results
    are returned; the command always exits 0 at this stage, even when every
    item was rejected.
    """
    items = load_batch_descriptor(file_path)

    # The trust phase runs exactly once, before any credential is read: a
    # credential can never influence admission or material selection.
    try:
        context = prepare_verifier_context(
            manifest_path, model, settings_path, vk_path, srs_path,
            registry_path, model_version)
    except (RegistryError, ZkError, OSError):
        raise BatchVerifyError(CODE_TRUST_CHECK_FAILED) from None

    results = [_verify_one(ident, credential_path, context)
               for ident, credential_path in items]
    accepted = sum(1 for result in results if result["accepted"])
    return {"total": len(items), "accepted": accepted,
            "rejected": len(items) - accepted, "results": results}
