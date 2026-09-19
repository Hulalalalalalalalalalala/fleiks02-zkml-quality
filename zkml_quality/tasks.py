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
import concurrent.futures
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
CODE_CAPACITY_EXCEEDED = "capacity_exceeded"
CODE_BATCH_NOT_FOUND = "batch_not_found"
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
    CODE_CAPACITY_EXCEEDED: "the batch would exceed the store's queued-task capacity",
    CODE_BATCH_NOT_FOUND: "no batch with this id exists in the store",
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
_BATCH_ID_RE = re.compile(r"pb-[0-9a-f]{24}")
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
_BATCH_KEYS = {"batch_id", "created_at", "task_ids"}
_BATCH_ITEM_REQUIRED = {"input", "model", "setup_dir", "credential"}
_BATCH_ITEM_OPTIONAL = {"idempotency_key"}
_DOCUMENT_REQUIRED = {"format_version", "kind", "tasks"}
_DOCUMENT_KEYS = _DOCUMENT_REQUIRED | {"batches"}


class TaskError(Exception):
    """Base class for task-store failures; carries a stable error code.

    ``retryable`` defaults to the class attribute, so categories that are
    always retryable (``store_io``) stay retryable even when raised without
    an explicit flag.
    """

    code = CODE_INVALID_REQUEST
    retryable = False

    def __init__(self, code=None, retryable=None):
        super().__init__(SAFE_MESSAGES[code or self.code])
        if code is not None:
            self.code = code
        if retryable is not None:
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


class TaskCapacityExceeded(TaskError):
    code = CODE_CAPACITY_EXCEEDED


class TaskBatchNotFound(TaskError):
    code = CODE_BATCH_NOT_FOUND


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


def _validate_batch(batch):
    if not isinstance(batch, dict) or set(batch) != _BATCH_KEYS:
        raise TaskStoreCorrupt()
    if not isinstance(batch["batch_id"], str) or not _BATCH_ID_RE.fullmatch(batch["batch_id"]):
        raise TaskStoreCorrupt()
    if not _is_timestamp(batch["created_at"]):
        raise TaskStoreCorrupt()
    task_ids = batch["task_ids"]
    if not isinstance(task_ids, list) or not task_ids:
        raise TaskStoreCorrupt()
    if any(not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id)
           for task_id in task_ids):
        raise TaskStoreCorrupt()


def _validate_document(document):
    if not isinstance(document, dict) \
            or not _DOCUMENT_REQUIRED <= set(document) \
            or not set(document) <= _DOCUMENT_KEYS:
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
    # Stores written before batches existed are still valid; the missing
    # section is normalised to an empty one in memory (and on the next write).
    batches = document.get("batches")
    if batches is None:
        document["batches"] = {}
        return
    if not isinstance(batches, dict):
        raise TaskStoreCorrupt()
    seen_batch_ids = set()
    for batch_id, batch in batches.items():
        _validate_batch(batch)
        if batch_id != batch["batch_id"] or batch_id in seen_batch_ids:
            raise TaskStoreCorrupt()
        seen_batch_ids.add(batch_id)
        task_ids = batch["task_ids"]
        if len(set(task_ids)) != len(task_ids):
            raise TaskStoreCorrupt()
        for task_id in task_ids:
            if task_id not in tasks:
                raise TaskStoreCorrupt()


def _empty_document():
    return {"format_version": FORMAT_VERSION, "kind": STORE_KIND,
            "tasks": {}, "batches": {}}


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
        _recover_credentials(document)

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


def _staging_path(credential_path, task_id, attempt_number):
    """The derivable staging path a proof attempt publishes its credential from.

    The name is deterministic so that, after a crash, recovery can find the
    staged credential of exactly one attempt without any extra journal.
    """
    credential_path = Path(credential_path)
    return credential_path.with_name(
        f".{credential_path.name}.{task_id}-{attempt_number}.pending")


def _recover_credentials(document):
    """Complete interrupted credential publishes of ``succeeded`` tasks.

    The ``succeeded`` state and the credential digest are committed to the
    store *before* the staged credential is renamed onto its final path, so a
    crash in that window is recoverable: if the final credential is missing or
    does not match the recorded digest but the staging file does, the publish
    is completed atomically. Anything unrecoverable is left untouched, and a
    task that never succeeded never had anything published to recover.
    """
    for task in document["tasks"].values():
        if task["state"] != STATE_SUCCEEDED:
            continue
        digest = task["credential_sha256"]
        final = Path(task["request"]["credential_path"])
        try:
            if final.is_file() and _sha256(final) == digest:
                continue
        except OSError:
            pass
        staging = _staging_path(final, task["task_id"], task["attempt"])
        try:
            if staging.is_file() and _sha256(staging) == digest:
                os.replace(staging, final)
        except OSError:
            pass


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

    All four paths are frozen to absolute canonical form against the calling
    directory at create time, so a later ``run`` from any other working
    directory still addresses exactly the artifacts that were validated.
    """
    input_path = Path(input_path).expanduser().resolve()
    model = Path(model).expanduser().resolve()
    setup_dir = Path(setup_dir).expanduser().resolve()
    credential_path = Path(credential_path).expanduser().resolve()
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


def _validate_idempotency_key(idempotency_key):
    if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not 1 <= len(idempotency_key) <= 200):
        raise TaskError(CODE_INVALID_REQUEST)


def _new_task_id(document):
    task_id = "pt-" + uuid.uuid4().hex[:24]
    while task_id in document["tasks"]:  # pragma: no cover - uuid4 collisions
        task_id = "pt-" + uuid.uuid4().hex[:24]
    return task_id


def _new_task(task_id, request, idempotency_key, timestamp):
    return {
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


def create_task(store_path, input_path, model, setup_dir, credential_path,
                idempotency_key=None):
    """Atomically create a task in ``queued``; honour an idempotency key.

    A repeated create with the same key and a byte-identical request returns
    the original task. The same key with a different request is rejected
    without touching the store.
    """
    _validate_idempotency_key(idempotency_key)
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
        task_id = _new_task_id(document)
        task = _new_task(task_id, request, idempotency_key, timestamp)
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
        digest = outcome.get("credential_sha256")
        if not _is_sha256_hex(digest):
            # The proof succeeded but its staged output vanished: never report
            # success without a credential.
            outcome.update(ok=False, code=CODE_OUTPUT_IO,
                           retryable=RETRYABLE[CODE_OUTPUT_IO])
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
    ``conflict``. The proof is written to a per-attempt staging file and the
    ``succeeded`` state (with the credential digest) is committed to the store
    *before* the credential is atomically renamed onto its final path, so a
    failure, interruption or crash can never overwrite a pre-existing
    credential, and a task that did not succeed never leaves a new credential
    behind. A crash between the commit and the rename is healed by
    ``_recover_credentials`` on the next store access.
    """
    # Claim under the lock, then prove without holding it.
    with _locked_store(store_path, create_parents=False) as (document, persist):
        task, attempt_number = _claim(document, task_id)
        request = task["request"]
        persist()

    credential_path = Path(request["credential_path"])
    staging_path = _staging_path(credential_path, task_id, attempt_number)
    outcome = None
    try:
        _verify_request_materials(request)
        features, paths, model_sha, scale, _final_path = _prepare_prove(
            request["input_path"], request["model_path"],
            request["setup_dir"], request["credential_path"])
        _issue_credential(features, paths, model_sha, scale, staging_path)
        try:
            digest = _sha256(staging_path)
        except OSError:
            outcome = {"ok": False, "code": CODE_OUTPUT_IO,
                       "retryable": RETRYABLE[CODE_OUTPUT_IO]}
        else:
            outcome = {"ok": True, "credential_sha256": digest}
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

    if not outcome["ok"]:
        # A task that did not succeed must not leave a new credential behind.
        with contextlib.suppress(OSError):
            staging_path.unlink()

    # Finalise under the lock from a fresh on-disk read. A failed proof is
    # recorded first; a failure of that bookkeeping (e.g. corrupt store) is a
    # TaskError that must not be masked by the proof error.
    with _locked_store(store_path, create_parents=False) as (document, persist):
        finished = _finish(document, task_id, attempt_number, outcome)
        persist()
        view = _safe_view(finished)
    if not outcome["ok"]:
        raise TaskAttemptFailed(outcome["code"], outcome["retryable"], view)
    try:
        # Publish only now that the succeeded state is durable. If this rename
        # fails (or the process dies here), recovery completes the publish.
        os.replace(staging_path, credential_path)
    except OSError:
        pass
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


# -- batches -----------------------------------------------------------------


def _parse_batch_items(items):
    """Validate the raw ``items`` array of a batch-create request.

    Every item takes exactly the single-task prove arguments plus an optional
    idempotency key; anything else rejects the whole batch before any proving
    pre-flight or store write happens.
    """
    if not isinstance(items, list) or not items:
        raise TaskError(CODE_INVALID_REQUEST)
    specs = []
    for item in items:
        if not isinstance(item, dict) \
                or not _BATCH_ITEM_REQUIRED <= set(item) \
                or not set(item) <= _BATCH_ITEM_REQUIRED | _BATCH_ITEM_OPTIONAL:
            raise TaskError(CODE_INVALID_REQUEST)
        for field in _BATCH_ITEM_REQUIRED:
            if not isinstance(item[field], str) or not item[field]:
                raise TaskError(CODE_INVALID_REQUEST)
        idempotency_key = item.get("idempotency_key")
        _validate_idempotency_key(idempotency_key)
        specs.append((item["input"], item["model"], item["setup_dir"],
                      item["credential"], idempotency_key))
    return specs


def _batch_view(batch, document):
    """Total, four-state counts and safe task views in creation order."""
    tasks = [document["tasks"][task_id] for task_id in batch["task_ids"]]
    states = {STATE_QUEUED: 0, STATE_RUNNING: 0,
              STATE_SUCCEEDED: 0, STATE_FAILED: 0}
    for task in tasks:
        states[task["state"]] += 1
    return {
        "batch_id": batch["batch_id"],
        "total": len(tasks),
        "states": states,
        "tasks": [_safe_view(task) for task in tasks],
    }


def _get_batch(document, batch_id):
    if not isinstance(batch_id, str) or not _BATCH_ID_RE.fullmatch(batch_id):
        raise TaskError(CODE_INVALID_REQUEST)
    batch = document["batches"].get(batch_id)
    if batch is None:
        raise TaskBatchNotFound()
    return batch


def create_batch(store_path, items, max_queued=None):
    """Atomically create a whole batch of queued tasks, or nothing at all.

    Every item is pre-validated exactly like a single ``create`` before
    anything is written; one bad item rejects the entire batch and the store
    is left untouched. The batch id is a stable function of the item
    requests, so re-creating the identical batch returns the original batch
    instead of duplicating its tasks. ``max_queued`` caps the total number of
    queued tasks in the store; exceeding it fails with the non-retryable
    ``capacity_exceeded`` and writes nothing.
    """
    if max_queued is not None and (
            not isinstance(max_queued, int) or isinstance(max_queued, bool)
            or max_queued < 0):
        raise TaskError(CODE_INVALID_REQUEST)
    specs = _parse_batch_items(items)
    prepared = []
    for input_path, model, setup_dir, credential_path, idempotency_key in specs:
        try:
            request = _build_request(input_path, model, setup_dir, credential_path)
        except TaskError:
            raise
        except ZkError as error:
            code, retryable = _classify(error)
            raise TaskError(code, retryable=retryable) from None
        prepared.append((idempotency_key, request))
    # A key repeated inside the batch must name the identical request; equal
    # (key, request) pairs collapse into a single task.
    units = []          # (idempotency_key, request), deduplicated by key
    unit_of_key = {}
    item_units = []     # unit index per item, in item order
    for idempotency_key, request in prepared:
        if idempotency_key is not None:
            previous = unit_of_key.get(idempotency_key)
            if previous is not None:
                if units[previous][1]["request_sha256"] != request["request_sha256"]:
                    raise TaskIdempotencyConflict()
                item_units.append(previous)
                continue
            unit_of_key[idempotency_key] = len(units)
        units.append((idempotency_key, request))
        item_units.append(len(units) - 1)
    batch_id = "pb-" + hashlib.sha256(_canonical_json(
        [[key, request["request_sha256"]] for key, request in prepared]
    ).encode("utf-8")).hexdigest()[:24]
    timestamp = _utcnow()
    with _locked_store(store_path) as (document, persist):
        existing = document["batches"].get(batch_id)
        if existing is not None:
            return _batch_view(existing, document), False
        keyed = {}
        for task in document["tasks"].values():
            if task["idempotency_key"] is not None:
                keyed[task["idempotency_key"]] = task
        unit_tasks = []     # task id per unit, None for tasks still to create
        for idempotency_key, request in units:
            if idempotency_key is not None and idempotency_key in keyed:
                existing_task = keyed[idempotency_key]
                if existing_task["request"]["request_sha256"] != request["request_sha256"]:
                    raise TaskIdempotencyConflict()
                unit_tasks.append(existing_task["task_id"])
            else:
                unit_tasks.append(None)
        new_count = sum(1 for task_id in unit_tasks if task_id is None)
        if max_queued is not None:
            queued = sum(1 for task in document["tasks"].values()
                         if task["state"] == STATE_QUEUED)
            if queued + new_count > max_queued:
                raise TaskCapacityExceeded()
        for index, (idempotency_key, request) in enumerate(units):
            if unit_tasks[index] is None:
                task_id = _new_task_id(document)
                document["tasks"][task_id] = _new_task(
                    task_id, request, idempotency_key, timestamp)
                unit_tasks[index] = task_id
        task_ids = list(dict.fromkeys(unit_tasks[index] for index in item_units))
        document["batches"][batch_id] = {
            "batch_id": batch_id,
            "created_at": timestamp,
            "task_ids": task_ids,
        }
        persist()
        return _batch_view(document["batches"][batch_id], document), True


def status_batch(store_path, batch_id):
    """Return the batch projection; never mutates the store."""
    with _locked_store(store_path, create_parents=False) as (document, _persist):
        return _batch_view(_get_batch(document, batch_id), document)


def run_batch(store_path, batch_id, max_workers=1):
    """Claim and execute the queued tasks of one batch, then report the batch.

    Only this batch's tasks are ever claimed, at most ``max_workers`` run
    concurrently, and each task goes through the same exclusive claim,
    attempt accounting and real EZKL proof as a single ``run``. A failing
    item is journaled on its task and never blocks the remaining items;
    tasks that are not queued (running, succeeded, failed) are left alone.
    """
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) \
            or max_workers < 1:
        raise TaskError(CODE_INVALID_REQUEST)
    with _locked_store(store_path, create_parents=False) as (document, _persist):
        batch = _get_batch(document, batch_id)
        task_ids = list(batch["task_ids"])

    def execute(task_id):
        try:
            run_task(store_path, task_id)
        except TaskAttemptFailed:
            pass  # the failed attempt is journaled on the task itself
        except (TaskInvalidState, TaskConflict):
            pass  # not claimable (already running/succeeded/failed); leave it

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="proof-task") as pool:
        list(pool.map(execute, task_ids))

    with _locked_store(store_path, create_parents=False) as (document, _persist):
        return _batch_view(_get_batch(document, batch_id), document)
