"""Offline batch verification of zk-quality credentials.

``zk-verify-batch`` is the batch form of ``zk-verify``. It reuses the exact
same seven verifier-owned trust parameters (registry, model version, trusted
manifest, model, settings, VK, SRS) and the same trust chain: the registry
admits one explicitly enabled version, the verifier's own manifest pins every
material digest, and the admitted registry record is compared with the
manifest field by field. That gate runs exactly once, *before* any credential
is opened; if it fails the whole batch is terminated and no item is verified.

The batch descriptor is a JSON file with exactly one top-level key, ``items``:
a non-empty array of at most 256 objects, each containing exactly ``id`` and
``credential``. Ids match ``[A-Za-z0-9._-]+`` and are unique within the batch;
duplicate JSON keys, unknown fields and any illegal structure reject the whole
descriptor before the trust gate or any verification runs.

After the gate, items are verified strictly in input order, each against the
same validated materials via the real offline EZKL verifier. A credential that
is absent, unreadable or structurally/semantically invalid, a digest or
summary mismatch, and a tampered proof or public output only reject *that
item*; they never stop the remaining items. The command therefore exits 0
even when items were rejected and prints exactly one JSON object to stdout
with ``total``, ``accepted``, ``rejected`` and ``results`` (input order).

Privacy mirrors ``zk-verify``: no output ever contains a file path, a feature
vector, proof bytes or an exception message. Failures of the descriptor or of
the trust materials are command-level failures: nonzero exit, empty stdout,
and exactly one fixed, non-revealing JSON object on stderr. Nothing is
written to disk and no network is contacted.
"""
import json
from pathlib import Path

from .registry import VERSION_RE, RegistryError
from .zk import (
    ZkError,
    _confirm_settings,
    _cross_check_credential,
    _load_credential,
    _run_ezkl_verify,
    _validate_proven_output,
    establish_verifier_trust,
)

MAX_ITEMS = 256

# Per-item rejection codes. ``credential_missing`` means the named credential
# cannot be opened; ``credential_invalid`` covers every structural, digest and
# summary failure before/around EZKL; ``verification_failed`` means EZKL ran
# and rejected the proof (or the proven instances do not match the summary).
CODE_CREDENTIAL_MISSING = "credential_missing"
CODE_CREDENTIAL_INVALID = "credential_invalid"
CODE_VERIFICATION_FAILED = "verification_failed"

# Command-level failure codes for the single stderr JSON object.
CODE_INVALID_BATCH = "invalid_batch"
CODE_TRUST_FAILED = "trust_verification_failed"
CODE_INTERNAL = "internal_error"

ITEM_MESSAGES = {
    CODE_CREDENTIAL_MISSING: "the credential could not be read",
    CODE_CREDENTIAL_INVALID: "the credential is invalid",
    CODE_VERIFICATION_FAILED: "the proof failed verification",
}
COMMAND_MESSAGES = {
    CODE_INVALID_BATCH: "the batch descriptor is invalid",
    CODE_TRUST_FAILED: "the verifier trust materials could not be established",
    CODE_INTERNAL: "the batch verification command failed",
}


class BatchError(Exception):
    """A command-level batch failure (bad descriptor or failed trust gate)."""

    def __init__(self, code):
        super().__init__(COMMAND_MESSAGES[code])
        self.code = code


def _reject_duplicate_keys(pairs):
    """json loader hook: any duplicated key makes the descriptor corrupt."""
    document = {}
    for key, value in pairs:
        if key in document:
            raise BatchError(CODE_INVALID_BATCH)
        document[key] = value
    return document


def load_batch_descriptor(file_path):
    """Strictly load and validate the batch descriptor.

    The top level must be an object containing only the non-empty ``items``
    array (1..256 entries). Each item must be an object containing only an
    ``id`` matching ``[A-Za-z0-9._-]+`` that is unique in the batch and a
    non-empty ``credential`` path string. Every deviation raises
    ``BatchError(invalid_batch)``; nothing about the credentials is consulted.
    """
    try:
        raw = Path(file_path).read_text(encoding="utf-8")
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  BatchError(CODE_INVALID_BATCH)))
    except BatchError:
        raise
    except (OSError, json.JSONDecodeError, ValueError, UnicodeError):
        raise BatchError(CODE_INVALID_BATCH) from None

    if not isinstance(document, dict) or set(document) != {"items"}:
        raise BatchError(CODE_INVALID_BATCH)
    items = document["items"]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise BatchError(CODE_INVALID_BATCH)

    validated = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "credential"}:
            raise BatchError(CODE_INVALID_BATCH)
        identifier = item["id"]
        credential = item["credential"]
        if not isinstance(identifier, str) or not VERSION_RE.fullmatch(identifier):
            raise BatchError(CODE_INVALID_BATCH)
        if identifier in seen:
            raise BatchError(CODE_INVALID_BATCH)
        if not isinstance(credential, str) or not credential:
            raise BatchError(CODE_INVALID_BATCH)
        seen.add(identifier)
        validated.append((identifier, credential))
    return validated


def _reject_item(code):
    return {"accepted": False, "error": {
        "code": code, "message": ITEM_MESSAGES[code]}}


def _verify_one(credential_path, materials, settings_scale):
    """Verify one credential against the already-trusted materials.

    All failures are mapped to the three stable per-item codes; no path,
    exception text or credential content is ever returned.
    """
    path = Path(credential_path)
    try:
        if not path.is_file():
            return _reject_item(CODE_CREDENTIAL_MISSING)
        credential = _load_credential(path)
    except OSError:
        return _reject_item(CODE_CREDENTIAL_MISSING)
    except (ZkError, ValueError, json.JSONDecodeError):
        return _reject_item(CODE_CREDENTIAL_INVALID)

    try:
        # Format, embedded digests/parameters and the public-output summary
        # are checked against the trusted material before EZKL runs. Any
        # disagreement is an invalid credential, not a failed proof.
        _cross_check_credential(credential, materials)
    except (ZkError, ValueError, TypeError):
        return _reject_item(CODE_CREDENTIAL_INVALID)

    try:
        _run_ezkl_verify(credential, materials)
        scores, fixed_point, label = _validate_proven_output(
            credential, settings_scale)
    except (ZkError, ValueError, json.JSONDecodeError, OSError):
        # A proof or public output that fails the real EZKL check, or whose
        # proven instances contradict the credential summary, is a failed
        # verification. The fixed message reveals neither which check failed
        # nor any proof content.
        return _reject_item(CODE_VERIFICATION_FAILED)

    return {"accepted": True, "model_sha256": materials["model_sha256"],
            "quantized_scores": scores, "scores_fixed_point": fixed_point,
            "label": label}


def verify_batch(file_path, manifest_path, model, settings_path, vk_path, srs_path,
                 registry_path, model_version):
    """Validate the descriptor, gate trust once, then verify items in order.

    Returns the single result document ``{total, accepted, rejected,
    results}``. A bad descriptor or a failed trust gate raises
    ``BatchError`` (the command must exit nonzero with empty stdout); item
    rejections are part of the normal result and never raise.
    """
    items = load_batch_descriptor(file_path)

    # One-time trust gate, identical to single zk-verify: admission, trusted
    # manifest/materials and the registry-record comparison. Any failure here
    # terminates the whole batch before the first credential is opened.
    try:
        _record, materials = establish_verifier_trust(
            registry_path, model_version,
            manifest_path, model, settings_path, vk_path, srs_path)
        settings_scale = _confirm_settings(materials)
    except (RegistryError, ZkError, OSError, ValueError) as error:
        raise BatchError(CODE_TRUST_FAILED) from None

    results = []
    accepted = 0
    for identifier, credential_path in items:
        outcome = _verify_one(credential_path, materials, settings_scale)
        entry = {"id": identifier}
        entry.update(outcome)
        if outcome["accepted"]:
            accepted += 1
        results.append(entry)

    return {"total": len(results), "accepted": accepted,
            "rejected": len(results) - accepted, "results": results}
