"""Offline, durable proof-task store around the real EZKL proving loop.

A proof task captures the same four arguments as ``zk-prove`` — the private
feature input, the ONNX model, the setup directory and the credential target —
and drives them through a strict lifecycle::

    queued -> running -> succeeded
    queued -> running -> failed (retryable or terminal)
    failed (retryable) -> queued   (via ``retry``)

The store is a single prover-owned JSON file next to a sibling ``.lock`` file.
Every state transition is published atomically (same-directory temp file plus
``os.replace``) while an exclusive ``flock`` serialises competing workers, so
exactly one executor can ever hold a task. The store is treated as hostile
input on every load: a missing, corrupt or structurally illegal store is
refused and never rewritten, and an illegal on-disk state is rejected rather
than "repaired".

Privacy: the private feature vector, the input document, the proof and every
file path are kept out of every command result; only non-sensitive digests are
emitted. The feature values are never persisted at all — the input file is
read again when the task actually runs — and a finished task records only the
SHA-256 of the credential it produced, never the credential itself. Paths are
held in the on-disk job request because the (local, offline) worker needs them
to perform the deferred proof; they never appear in any command's output.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .zk import (
    ZkArtifactError,
    ZkArtifactMissing,
    ZkDigestMismatch,
    ZkError,
    ZkEzklError,
    ZkInputError,
    ZkOutputError,
    _issue_credential,
    _prepare_prove,
    _read_manifest,
    _sha256,
)

FORMAT_VERSION = 1
STORE_KIND = "zk-quality-proof-tasks"

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
TERMINAL_ATTEMPT_STATES = (STATE_RUNNING, STATE_SUCCEEDED, STATE_FAILED)

# Error codes surfaced on stderr. The first six are the failure categories the
# command line must distinguish; the remainder describe task-store control
# failures where no proof attempt was made.
CODE_INVALID_REQUEST = "invalid_request"
CODE_INVALID_INPUT = "invalid_input"
CODE_ARTIFACT_MISSING = "artifact_missing"
CODE_DIGEST_MISMATCH = "digest_mismatch"
CODE_EZKL_FAILURE = "ezkl_failure"
CODE_OUTPUT_IO = "output_io_failure"
CODE_INTERRUPTED = "interrupted"
CODE_TASK_NOT_FOUND = "task_not_found"
CODE_INVALID_STATE = "invalid_state"
CODE_CONFLICT = "conflict"
CODE_IDEMPOTENCY_CONFLICT = "idempotency_conflict"
CODE_STORE_CORRUPT = "store_corrupt"
CODE_STORE_IO = "store_io"

# Fixed, non-revealing messages: the message never embeds exception text,
# paths or input content, so a failure cannot leak the job arguments.
SAFE_MESSAGES = {
    CODE_INVALID_REQUEST: "the proof-task request is invalid",
    CODE_INVALID_INPUT: "the proof input is invalid",
    CODE_ARTIFACT_MISSING: "a required proof artifact is missing or unreadable",
    CODE_DIGEST_MISMATCH: "an artifact does not match its pinned digest",
    CODE_EZKL_FAILURE: "proof generation failed",
    CODE_OUTPUT_IO: "the credential could not be written",
    CODE_INTERRUPTED: "the proof task was interrupted",
    CODE_TASK_NOT_FOUND: "no task with this id exists in the store",
    CODE_INVALID_STATE: "the task is not in a state that allows this operation",
    CODE_CONFLICT: "another executor already holds the task",
    CODE_IDEMPOTENCY_CONFLICT: "the idempotency key is already used by a different request",
    CODE_STORE_CORRUPT: "the task store is corrupt or in an illegal state",
    CODE_STORE_IO: "the task store could not be read or written",
}

# Whether re-enqueueing the task could plausibly help. Control failures never
# consume an attempt and carry their own (non-retryable) default below.
RETRYABLE = {
    CODE_INVALID_REQUEST: False,
    CODE_INVALID_INPUT: False,
    CODE_ARTIFACT_MISSING: True,
    CODE_DIGEST_MISMATCH: False,
    CODE_EZKL_FAILURE: True,
    CODE_OUTPUT_IO: True,
    CODE_INTERRUPTED: True,
}

_HEX64 = set("0123456789abcdef")
_TASK_ID_RE = re.compile(r"pt-[0-9a-f]{24}")
_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")

_SETUP_REQUEST_KEYS = {"ezkl_version", "logrows", "output_scale", "manifest_sha256", "artifacts"}
_REQUEST_KEYS = {
    "request_sha256", "input_path", "input_sha256",
    "model_path", "model_sha256", "setup_dir", "credential_path", "setup",
}
_TASK_KEYS = {
    "task_id", "state", "attempt", "created_at", "updated_at",
    "started_at", "ended_at", "idempotency_key", "request",
    "attempts", "credential_sha256", "last_error",
}
_ATTEMPT_KEYS = {
    "attempt", "state", "started_at", "ended_at",
    "retryable", "error_code", "credential_sha256",
}
_SETUP_ARTIFACT_KEYS = {"compiled", "settings", "pk", "vk", "srs"}


class TaskError(Exception):
    """Base class for task-store failures; carries a stable error code."""

    code = CODE_INVALID_REQUEST
    retryable = False

    def __init__(self, code=None, retryable=False):
        super().__init__(SAFE_MESSAGES[code or self.code])
        if code is not None:
            self.code = code
        self.retryable = retryable


class TaskAttemptFailed(TaskError):
    """A recorded failed proof attempt; the store transition already happened."""

    def __init__(self, code, retryable, view):
        super().__init__(code, retryable=retryable)
        self.view = view


class TaskNotFound(TaskError):
    code = CODE_TASK_NOT_FOUND


class TaskInvalidState(TaskError):
    code = CODE_INVALID_STATE


class TaskConflict(TaskError):
    code = CODE_CONFLICT


class TaskIdempotencyConflict(TaskError):
    code = CODE_IDEMPOTENCY_CONFLICT


class TaskStoreCorrupt(TaskError):
    code = CODE_STORE_CORRUPT


class TaskStoreIo(TaskError):
    code = CODE_STORE_IO
    retryable = True


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _is_sha256_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX64 for char in value)


def _is_timestamp(value):
    return isinstance(value, str) and bool(_TS_RE.fullmatch(value))


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise TaskStoreCorrupt()
        document[key] = value
    return document


def _validate_attempt(entry):
    if not isinstance(entry, dict) or set(entry) != _ATTEMPT_KEYS:
        raise TaskStoreCorrupt()
    attempt = entry["attempt"]
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise TaskStoreCorrupt()
    if entry["state"] not in TERMINAL_ATTEMPT_STATES:
        raise TaskStoreCorrupt()
    if not _is_timestamp(entry["started_at"]):
        raise TaskStoreCorrupt()
    if entry["state"] == STATE_RUNNING:
        if entry["ended_at"] is not None:
            raise TaskStoreCorrupt()
    elif not _is_timestamp(entry["ended_at"]):
        raise TaskStoreCorrupt()
    if not isinstance(entry["retryable"], (bool, type(None))):
        raise TaskStoreCorrupt()
    if entry["state"] == STATE_FAILED:
        if not isinstance(entry["retryable"], bool):
            raise TaskStoreCorrupt()
        if not isinstance(entry["error_code"], str) or not entry["error_code"]:
            raise TaskStoreCorrupt()
    elif entry["error_code"] is not None:
        raise TaskStoreCorrupt()
    if entry["state"] == STATE_SUCCEEDED:
        if not _is_sha256_hex(entry["credential_sha256"]):
            raise TaskStoreCorrupt()
    elif entry["credential_sha256"] is not None:
        raise TaskStoreCorrupt()


def _validate_request(request):
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS:
        raise TaskStoreCorrupt()
    if not _is_sha256_hex(request["request_sha256"]) \
            or not _is_sha256_hex(request["input_sha256"]) \
            or not _is_sha256_hex(request["model_sha256"]):
        raise TaskStoreCorrupt()
    for key in ("input_path", "model_path", "setup_dir", "credential_path"):
        if not isinstance(request[key], str) or not request[key]:
            raise TaskStoreCorrupt()
    setup = request["setup"]
    if not isinstance(setup, dict) or set(setup) != _SETUP_REQUEST_KEYS:
        raise TaskStoreCorrupt()
    if not isinstance(setup["ezkl_version"], str) or not setup["ezkl_version"]:
        raise TaskStoreCorrupt()
    if not isinstance(setup["logrows"], int) or isinstance(setup["logrows"], bool) \
            or not 1 <= setup["logrows"] <= 30:
        raise TaskStoreCorrupt()
    if not isinstance(setup["output_scale"], int) or isinstance(setup["output_scale"], bool) \
            or setup["output_scale"] <= 0:
        raise TaskStoreCorrupt()
    if not _is_sha256_hex(setup["manifest_sha256"]):
        raise TaskStoreCorrupt()
    artifacts = setup["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != _SETUP_ARTIFACT_KEYS:
        raise TaskStoreCorrupt()
    if any(not _is_sha256_hex(artifacts[key]) for key in _SETUP_ARTIFACT_KEYS):
        raise TaskStoreCorrupt()
    if _request_seal_hash(request) != request["request_sha256"]:
        raise TaskStoreCorrupt()


def _validate_task(task):
    if not isinstance(task, dict) or set(task) != _TASK_KEYS:
        raise TaskStoreCorrupt()
    if not isinstance(task["task_id"], str) or not _TASK_ID_RE.fullmatch(task["task_id"]):
        raise TaskStoreCorrupt()
    if task["state"] not in (STATE_QUEUED, STATE_RUNNING, STATE_SUCCEEDED, STATE_FAILED):
        raise TaskStoreCorrupt()
    attempt = task["attempt"]
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise TaskStoreCorrupt()
    if not _is_timestamp(task["created_at"]) or not _is_timestamp(task["updated_at"]):
        raise TaskStoreCorrupt()
    if task["idempotency_key"] is not None and (
            not isinstance(task["idempotency_key"], str)
            or not 1 <= len(task["idempotency_key"]) <= 200):
        raise TaskStoreCorrupt()
    _validate_request(task["request"])
    attempts = task["attempts"]
    if not isinstance(attempts, list) or len(attempts) != attempt:
        raise TaskStoreCorrupt()
    for index, entry in enumerate(attempts, start=1):
        _validate_attempt(entry)
        if entry["attempt"] != index:
            raise TaskStoreCorrupt()

    last = attempts[-1] if attempts else None
    if task["state"] == STATE_QUEUED:
        if task["started_at"] is not None or task["ended_at"] is not None:
            raise TaskStoreCorrupt()
        if attempt == 0:
            if task["last_error"] is not None:
                raise TaskStoreCorrupt()
        elif last["state"] != STATE_FAILED or last["retryable"] is not True:
            raise TaskStoreCorrupt()
    elif task["state"] == STATE_RUNNING:
        if last is None or last["state"] != STATE_RUNNING:
            raise TaskStoreCorrupt()
        if not _is_timestamp(task["started_at"]) or task["ended_at"] is not None:
            raise TaskStoreCorrupt()
        if task["credential_sha256"] is not None or task["last_error"] is not None:
            raise TaskStoreCorrupt()
    else:
        if last is None or last["state"] != task["state"]:
            raise TaskStoreCorrupt()
        if not _is_timestamp(task["started_at"]) or not _is_timestamp(task["ended_at"]):
            raise TaskStoreCorrupt()
    if task["state"] == STATE_SUCCEEDED:
        if not _is_sha256_hex(task["credential_sha256"]) \
                or task["credential_sha256"] != last["credential_sha256"] \
                or task["last_error"] is not None:
            raise TaskStoreCorrupt()
    elif task["credential_sha256"] is not None:
        raise TaskStoreCorrupt()
    if task["state"] == STATE_FAILED:
        error = task["last_error"]
        if not isinstance(error, dict) or set(error) != {"code", "retryable", "message"} \
                or not isinstance(error["code"], str) or not error["code"] \
                or not isinstance(error["retryable"], bool) \
                or not isinstance(error["message"], str) or not error["message"] \
                or error["retryable"] != last["retryable"]:
            raise TaskStoreCorrupt()


def _validate_document(document):
    if not isinstance(document, dict) or set(document) != {"format_version", "kind", "tasks"}:
        raise TaskStoreCorrupt()
    if document["format_version"] != FORMAT_VERSION or document["kind"] != STORE_KIND:
        raise TaskStoreCorrupt()
    tasks = document["tasks"]
    if not isinstance(tasks, dict):
        raise TaskStoreCorrupt()
    seen_ids = set()
    seen_keys = set()
    for task_id, task in tasks.items():
        _validate_task(task)
        if task_id != task["task_id"] or task_id in seen_ids:
            raise TaskStoreCorrupt()
        seen_ids.add(task_id)
        key = task["idempotency_key"]
        if key is not None:
            if key in seen_keys:
                raise TaskStoreCorrupt()
            seen_keys.add(key)


def _empty_document():
    return {"format_version": FORMAT_VERSION, "kind": STORE_KIND, "tasks": {}}


def _load_document(store_path):
    try:
        raw = store_path.read_text(encoding="utf-8")
        document = json.loads(raw, object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  TaskStoreCorrupt()))
    except TaskStoreCorrupt:
        raise
    except OSError as error:
        raise TaskStoreIo() from error
    except (json.JSONDecodeError, ValueError):
        raise TaskStoreCorrupt()
    _validate_document(document)
    return document


def _atomic_write(store_path, document):
    """Validate then publish the document via a same-directory temp file."""
    _validate_document(document)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=store_path.parent,
        prefix=".proof-tasks-", suffix=".tmp", delete=False)
    try:
        json.dump(document, handle, ensure_ascii=False, allow_nan=False,
                  indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, store_path)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


@contextlib.contextmanager
def _locked_store(store_path, *, create_parents=True):
    """Yield ``(document, persist)`` with an exclusive lock on the store.

    Reads happen under the lock so a worker always observes the last committed
    transition. ``persist`` re-validates and atomically replaces the file; on
    any error the previous file is left byte-for-byte intact.

    Read/claim operations pass ``create_parents=False``: a store that does not
    exist at all is presented as empty without creating files or a lock, and
    attempting to persist such a view fails. This keeps ``status``/``run`` on
    a missing store a plain "task not found" rather than an I/O error.
    """
    store_path = Path(store_path)
    if store_path.is_dir():
        raise TaskStoreCorrupt()
    if not create_parents and not store_path.exists():
        if store_path.parent.exists() and not store_path.parent.is_dir():
            raise TaskStoreIo()

        def _persist_missing():
            raise TaskStoreIo()

        yield _empty_document(), _persist_missing
        return
    if store_path.parent.exists() and not store_path.parent.is_dir():
        raise TaskStoreIo()
    if create_parents:
        try:
            store_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TaskStoreIo() from error
    lock_path = store_path.with_name(store_path.name + ".lock")
    try:
        lock_handle = open(lock_path, "a+b")
    except OSError as error:
        raise TaskStoreIo() from error
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        document = _load_document(store_path) if store_path.exists() else _empty_document()

        def persist():
            try:
                _atomic_write(store_path, document)
            except TaskStoreCorrupt:
                raise
            except OSError as error:
                raise TaskStoreIo() from error

        yield document, persist
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _safe_view(task):
    """The only projection of a task that may leave the process."""
    return {
        "task_id": task["task_id"],
        "state": task["state"],
        "attempt": task["attempt"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "ended_at": task["ended_at"],
        "updated_at": task["updated_at"],
        "credential_sha256": task["credential_sha256"],
    }


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _request_seal_hash(request):
    sealed = {key: request[key] for key in (
        "input_path", "input_sha256", "model_path", "model_sha256",
        "setup_dir", "credential_path", "setup")}
    return hashlib.sha256(_canonical_json(sealed).encode("utf-8")).hexdigest()


def _build_request(input_path, model, setup_dir, credential_path):
    """Validate a prove request exactly like zk-prove and seal its facts.

    Returns the request record to persist. Feature values are validated but
    returned separately so the caller can discard them; they are never stored.
    """
    input_path = Path(input_path)
    model = Path(model)
    setup_dir = Path(setup_dir)
    credential_path = Path(credential_path)
    # Full zk-prove pre-flight: features, manifest, model and every setup
    # artifact are validated now, before a task id exists.
    features, _paths, model_sha, _scale, _cred = _prepare_prove(
        input_path, model, setup_dir, credential_path)
    del features
    try:
        manifest = _read_manifest(setup_dir / "manifest.json")
    except ZkError as error:
        raise ZkArtifactError(str(error)) from error

    def digest_of(path):
        try:
            return _sha256(path)
        except OSError as error:
            raise ZkArtifactError(f"artifact cannot be read: {error.strerror or error}") from error

    request = {
        "input_path": str(input_path),
        "input_sha256": digest_of(input_path),
        "model_path": str(model),
        "model_sha256": model_sha,
        "setup_dir": str(setup_dir),
        "credential_path": str(credential_path),
        "setup": {
            "ezkl_version": manifest["ezkl_version"],
            "logrows": int(manifest["logrows"]),
            "output_scale": int(manifest["output_scale"]),
            "manifest_sha256": digest_of(setup_dir / "manifest.json"),
            "artifacts": {key: manifest["artifacts"][key]
                          for key in sorted(_SETUP_ARTIFACT_KEYS)},
        },
    }
    request["request_sha256"] = _request_seal_hash(request)
    return request


def create_task(store_path, input_path, model, setup_dir, credential_path,
                idempotency_key=None):
    """Atomically create a task in ``queued``; honour an idempotency key.

    A repeated create with the same key and a byte-identical request returns
    the original task. The same key with a different request is rejected
    without touching the store.
    """
    if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not 1 <= len(idempotency_key) <= 200):
        raise TaskError(CODE_INVALID_REQUEST)
    try:
        request = _build_request(input_path, model, setup_dir, credential_path)
    except TaskError:
        raise
    except ZkError as error:
        # Classify (and thereby sanitise) every create-time proving error so a
        # rejected create reports a stable code and never echoes a path.
        code, retryable = _classify(error)
        raise TaskError(code, retryable=retryable) from None
    timestamp = _utcnow()
    with _locked_store(store_path) as (document, persist):
        if idempotency_key is not None:
            for existing in document["tasks"].values():
                if existing["idempotency_key"] == idempotency_key:
                    if existing["request"]["request_sha256"] == request["request_sha256"]:
                        return _safe_view(existing), False
                    raise TaskIdempotencyConflict()
        task_id = "pt-" + uuid.uuid4().hex[:24]
        while task_id in document["tasks"]:  # pragma: no cover - uuid4 collisions
            task_id = "pt-" + uuid.uuid4().hex[:24]
        task = {
            "task_id": task_id,
            "state": STATE_QUEUED,
            "attempt": 0,
            "created_at": timestamp,
            "updated_at": timestamp,
            "started_at": None,
            "ended_at": None,
            "idempotency_key": idempotency_key,
            "request": request,
            "attempts": [],
            "credential_sha256": None,
            "last_error": None,
        }
        document["tasks"][task_id] = task
        persist()
        return _safe_view(task), True


def _get_task(document, task_id):
    if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
        raise TaskError(CODE_INVALID_REQUEST)
    task = document["tasks"].get(task_id)
    if task is None:
        raise TaskNotFound()
    return task


def status_task(store_path, task_id):
    """Return the safe projection of one task; never mutates the store."""
    with _locked_store(store_path, create_parents=False) as (document, _persist):
        return _safe_view(_get_task(document, task_id))


def _claim(document, task_id):
    task = _get_task(document, task_id)
    if task["state"] == STATE_RUNNING:
        raise TaskConflict()
    if task["state"] != STATE_QUEUED:
        raise TaskInvalidState()
    timestamp = _utcnow()
    attempt_number = task["attempt"] + 1
    task["state"] = STATE_RUNNING
    task["attempt"] = attempt_number
    task["started_at"] = timestamp
    task["ended_at"] = None
    task["updated_at"] = timestamp
    task["credential_sha256"] = None
    task["last_error"] = None
    task["attempts"].append({
        "attempt": attempt_number,
        "state": STATE_RUNNING,
        "started_at": timestamp,
        "ended_at": None,
        "retryable": None,
        "error_code": None,
        "credential_sha256": None,
    })
    return task, attempt_number


def _finish(document, task_id, attempt_number, outcome):
    """Commit a terminal attempt state, but only if we still own the task."""
    task = _get_task(document, task_id)
    if task["state"] != STATE_RUNNING or task["attempt"] != attempt_number:
        raise TaskConflict()
    entry = task["attempts"][attempt_number - 1]
    if entry["state"] != STATE_RUNNING or entry["attempt"] != attempt_number:
        raise TaskStoreCorrupt()
    timestamp = _utcnow()
    entry["ended_at"] = timestamp
    task["ended_at"] = timestamp
    task["updated_at"] = timestamp
    if outcome["ok"]:
        credential_path = Path(task["request"]["credential_path"])
        if not credential_path.is_file():
            # The proof succeeded but its output vanished: never report
            # success without a credential.
            outcome.update(ok=False, code=CODE_OUTPUT_IO, retryable=True)
        else:
            try:
                digest = _sha256(credential_path)
            except OSError:
                outcome.update(ok=False, code=CODE_OUTPUT_IO, retryable=True)
            else:
                entry["state"] = STATE_SUCCEEDED
                entry["credential_sha256"] = digest
                task["state"] = STATE_SUCCEEDED
                task["credential_sha256"] = digest
    if not outcome["ok"]:
        code = outcome["code"]
        retryable = outcome["retryable"]
        entry["state"] = STATE_FAILED
        entry["retryable"] = retryable
        entry["error_code"] = code
        task["state"] = STATE_FAILED
        task["credential_sha256"] = None
        task["last_error"] = {
            "code": code, "retryable": retryable,
            "message": SAFE_MESSAGES[code]}
    return task


def _classify(error):
    """Map a proving exception to a stable code and retryable flag."""
    if isinstance(error, ZkInputError):
        return CODE_INVALID_INPUT, RETRYABLE[CODE_INVALID_INPUT]
    if isinstance(error, ZkArtifactMissing):
        return CODE_ARTIFACT_MISSING, RETRYABLE[CODE_ARTIFACT_MISSING]
    if isinstance(error, ZkArtifactError):
        # Present but unreadable: treat the artifact as unavailable; rerunning
        # after it is repaired is allowed.
        return CODE_ARTIFACT_MISSING, RETRYABLE[CODE_ARTIFACT_MISSING]
    if isinstance(error, ZkDigestMismatch):
        return CODE_DIGEST_MISMATCH, RETRYABLE[CODE_DIGEST_MISMATCH]
    if isinstance(error, ZkOutputError):
        return CODE_OUTPUT_IO, RETRYABLE[CODE_OUTPUT_IO]
    if isinstance(error, ZkEzklError):
        return CODE_EZKL_FAILURE, RETRYABLE[CODE_EZKL_FAILURE]
    if isinstance(error, TaskError):
        raise
    # Anything else escaping the proving stage is a proving-stage failure.
    return CODE_EZKL_FAILURE, RETRYABLE[CODE_EZKL_FAILURE]


def _verify_request_materials(request):
    """Confirm the on-disk job inputs are still exactly those sealed at create.

    The request's digests are a tamper-evident seal: a model, input, manifest
    or any setup artifact changed (or swapped between setups) between
    ``create`` and ``run`` is a digest mismatch and fails the attempt without
    ever invoking EZKL.
    """
    input_path = Path(request["input_path"])
    model_path = Path(request["model_path"])
    setup_dir = Path(request["setup_dir"])
    sealed = request["setup"]

    def digest_of(path):
        try:
            return _sha256(path)
        except OSError as error:
            raise ZkArtifactError(f"{path} cannot be read") from error

    if not input_path.is_file():
        raise ZkArtifactMissing("input file not found")
    if not model_path.is_file():
        raise ZkArtifactMissing("ONNX model not found")
    if digest_of(input_path) != request["input_sha256"]:
        raise ZkDigestMismatch("input changed since the task was created")
    if digest_of(model_path) != request["model_sha256"]:
        raise ZkDigestMismatch("model changed since the task was created")
    manifest_path = setup_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ZkArtifactMissing("setup manifest not found")
    if digest_of(manifest_path) != sealed["manifest_sha256"]:
        raise ZkDigestMismatch("setup manifest changed since the task was created")
    try:
        manifest = _read_manifest(manifest_path)
    except ZkArtifactMissing:
        raise
    except ZkError as error:
        raise ZkArtifactError(str(error)) from error
    for key in _SETUP_ARTIFACT_KEYS:
        filename = {
            "compiled": "compiled.ezkl",
            "settings": "settings.json",
            "pk": "proving.key",
            "vk": "verification.key",
            "srs": "srs",
        }[key]
        artifact = setup_dir / filename
        if not artifact.is_file():
            raise ZkArtifactMissing(f"setup artifact '{filename}' not found")
        if digest_of(artifact) != sealed["artifacts"][key]:
            raise ZkDigestMismatch(f"setup artifact '{filename}' changed")
    if _request_seal_hash(request) != request["request_sha256"]:
        raise ZkDigestMismatch("stored request seal is invalid")


def run_task(store_path, task_id):
    """Claim a queued task, run the real EZKL proof, and record the outcome.

    Only ``queued -> running -> succeeded/failed`` is possible. The claim is
    committed before proving starts, so competing runners fail with
    ``conflict``; success is recorded only after a real credential exists and
    is hashed. Every failure — including interruption — is recorded as a
    failed attempt and can never overwrite a prior credential.
    """
    # Claim under the lock, then prove without holding it.
    with _locked_store(store_path, create_parents=False) as (document, persist):
        task, attempt_number = _claim(document, task_id)
        request = task["request"]
        persist()

    outcome = None
    try:
        _verify_request_materials(request)
        features, paths, model_sha, scale, credential_path = _prepare_prove(
            request["input_path"], request["model_path"],
            request["setup_dir"], request["credential_path"])
        _issue_credential(features, paths, model_sha, scale, credential_path)
        outcome = {"ok": True}
    except BaseException as error:  # noqa: BLE001 - includes KeyboardInterrupt
        if isinstance(error, TaskError):
            raise
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            code, retryable = CODE_INTERRUPTED, RETRYABLE[CODE_INTERRUPTED]
        elif isinstance(error, OSError):
            code, retryable = CODE_OUTPUT_IO, RETRYABLE[CODE_OUTPUT_IO]
        else:
            code, retryable = _classify(error)
        outcome = {"ok": False, "code": code, "retryable": retryable}

    # Finalise under the lock from a fresh on-disk read. A failed proof is
    # recorded first; a failure of that bookkeeping (e.g. corrupt store) is a
    # TaskError that must not be masked by the proof error.
    with _locked_store(store_path, create_parents=False) as (document, persist):
        finished = _finish(document, task_id, attempt_number, outcome)
        persist()
        view = _safe_view(finished)
    if not outcome["ok"]:
        raise TaskAttemptFailed(outcome["code"], outcome["retryable"], view)
    return view


def retry_task(store_path, task_id):
    """Re-queue only a ``failed`` task whose last attempt was marked retryable.

    Succeeded, queued, running and terminal-failed tasks are all rejected; the
    full attempt history is preserved and the next run increments the attempt
    counter again.
    """
    with _locked_store(store_path, create_parents=False) as (document, persist):
        task = _get_task(document, task_id)
        if task["state"] != STATE_FAILED:
            raise TaskInvalidState()
        last = task["attempts"][-1]
        if last["retryable"] is not True:
            raise TaskInvalidState()
        timestamp = _utcnow()
        task["state"] = STATE_QUEUED
        task["started_at"] = None
        task["ended_at"] = None
        task["updated_at"] = timestamp
        task["credential_sha256"] = None
        # last_error is retained as history on the failed attempt entry; the
        # task-level pointer is cleared for the fresh execution.
        task["last_error"] = None
        persist()
        return _safe_view(task)
