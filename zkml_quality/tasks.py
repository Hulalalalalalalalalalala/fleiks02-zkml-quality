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
import errno
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
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

_BATCH_ID_RE = re.compile(r"pb-[0-9a-f]{24}")

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
CODE_CAPACITY_EXCEEDED = "capacity_exceeded"
CODE_BATCH_NOT_FOUND = "batch_not_found"

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
    CODE_CAPACITY_EXCEEDED: "the task store has reached its queued-task limit",
    CODE_BATCH_NOT_FOUND: "no batch with this id exists in the store",
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
    "pending_credential", "batch_id", "sequence",
}
_ATTEMPT_KEYS = {
    "attempt", "state", "started_at", "ended_at",
    "retryable", "error_code", "credential_sha256",
}
_SETUP_ARTIFACT_KEYS = {"compiled", "settings", "pk", "vk", "srs"}

# The four task states reported in batch totals.
BATCH_STATES = (STATE_QUEUED, STATE_RUNNING, STATE_SUCCEEDED, STATE_FAILED)

# Suffix used for the credential staging file. It lives next to the final
# credential so the publish is a same-directory atomic rename, and it is the
# only "new credential" file a non-succeeded task may leave behind.
_CRED_STAGING_SUFFIX = ".proof-task.tmp"

_PENDING_CREDENTIAL_KEYS = {"temp_path", "credential_sha256"}
_BATCH_KEYS = {"batch_id", "created_at", "sequence"}


# Attempts genuinely live in *this* process (file locks are shared across its
# threads, so an flock alone cannot fence a sibling thread). Recovery skips any
# running attempt whose key is present here; the flock handles a separate
# process. Keyed by (real store path, task id, attempt number).
_LIVE_ATTEMPTS = set()
_LIVE_ATTEMPTS_GUARD = threading.Lock()


def _attempt_lock_path(store_path, task_id, attempt_number):
    """Sibling file whose flock fences one attempt for its whole lifetime.

    The runner holds an exclusive lock from claim until the terminal state is
    persisted; the kernel releases it automatically when the process dies.
    Crash recovery takes over a journaled publish only when it can acquire
    this lock *and* the attempt is not live in the current process — i.e. the
    owning executor is genuinely gone. A merely slow live worker (in another
    process or another thread here) is never preempted.
    """
    store_path = Path(store_path)
    return store_path.with_name(
        f".{store_path.name}.{task_id}.{attempt_number}.lock")


def _attempt_key(store_path, task_id, attempt_number):
    return (os.path.realpath(str(store_path)), task_id, attempt_number)


def _attempt_is_live_locally(store_path, task_id, attempt_number):
    with _LIVE_ATTEMPTS_GUARD:
        return _attempt_key(store_path, task_id, attempt_number) in _LIVE_ATTEMPTS


def _acquire_attempt_lock(store_path, task_id, attempt_number):
    """Take the exclusive, process-registered fence for one claimed attempt."""
    lock_path = _attempt_lock_path(store_path, task_id, attempt_number)
    key = _attempt_key(store_path, task_id, attempt_number)
    try:
        handle = open(lock_path, "a+b")
    except OSError as error:
        raise TaskStoreIo() from error
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        if error.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
            raise TaskConflict()
        raise TaskStoreIo() from error
    with _LIVE_ATTEMPTS_GUARD:
        _LIVE_ATTEMPTS.add(key)
    return handle


def _release_attempt_lock(handle, store_path, task_id, attempt_number):
    """Release the attempt fence, unregister it, and best-effort remove file."""
    if handle is None:
        return
    key = _attempt_key(store_path, task_id, attempt_number)
    lock_path = _attempt_lock_path(store_path, task_id, attempt_number)
    with _LIVE_ATTEMPTS_GUARD:
        _LIVE_ATTEMPTS.discard(key)
    with contextlib.suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    with contextlib.suppress(OSError):
        lock_path.unlink()


class TaskError(Exception):
    """Base class for task-store failures; carries a stable error code."""

    code = CODE_INVALID_REQUEST
    retryable = False

    def __init__(self, code=None, retryable=None):
        super().__init__(SAFE_MESSAGES[code or self.code])
        if code is not None:
            self.code = code
        # Default to the class's declared retryability (e.g. TaskStoreIo is
        # always retryable); an explicit argument still overrides it.
        self.retryable = type(self).retryable if retryable is None else retryable


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


class TaskCapacityExceeded(TaskError):
    code = CODE_CAPACITY_EXCEEDED


class TaskBatchNotFound(TaskError):
    code = CODE_BATCH_NOT_FOUND


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


def _validate_pending_credential(pending):
    """Shape of the in-flight credential publish journaled on a running task."""
    if not isinstance(pending, dict) or set(pending) != _PENDING_CREDENTIAL_KEYS:
        raise TaskStoreCorrupt()
    if not isinstance(pending["temp_path"], str) or not pending["temp_path"]:
        raise TaskStoreCorrupt()
    if not _is_sha256_hex(pending["credential_sha256"]):
        raise TaskStoreCorrupt()


def _validate_batch_summary(batch):
    if not isinstance(batch, dict) or set(batch) != _BATCH_KEYS:
        raise TaskStoreCorrupt()
    if not isinstance(batch["batch_id"], str) or not _BATCH_ID_RE.fullmatch(batch["batch_id"]):
        raise TaskStoreCorrupt()
    if not _is_timestamp(batch["created_at"]):
        raise TaskStoreCorrupt()
    if not isinstance(batch["sequence"], int) or isinstance(batch["sequence"], bool) \
            or batch["sequence"] < 0:
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
    if task["batch_id"] is not None and (
            not isinstance(task["batch_id"], str)
            or not _BATCH_ID_RE.fullmatch(task["batch_id"])):
        raise TaskStoreCorrupt()
    if task["batch_id"] is None and task["sequence"] != 0:
        raise TaskStoreCorrupt()
    if not isinstance(task["sequence"], int) or isinstance(task["sequence"], bool) \
            or task["sequence"] < 0:
        raise TaskStoreCorrupt()
    _validate_request(task["request"])
    attempts = task["attempts"]
    if not isinstance(attempts, list) or len(attempts) != attempt:
        raise TaskStoreCorrupt()
    for index, entry in enumerate(attempts, start=1):
        _validate_attempt(entry)
        if entry["attempt"] != index:
            raise TaskStoreCorrupt()

    pending = task["pending_credential"]
    if pending is not None:
        _validate_pending_credential(pending)
        # The journal may only ever name the deterministic sibling staging
        # file derived from this task's own credential target; otherwise a
        # hostile store could steer recovery at an unrelated file.
        expected_staging = str(_pending_staging_path(
            Path(task["request"]["credential_path"])))
        if pending["temp_path"] != expected_staging:
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
        if pending is not None:
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
        if pending is not None:
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
    if not isinstance(document, dict) \
            or set(document) != {"format_version", "kind", "tasks", "batches"}:
        raise TaskStoreCorrupt()
    if document["format_version"] != FORMAT_VERSION or document["kind"] != STORE_KIND:
        raise TaskStoreCorrupt()
    tasks = document["tasks"]
    batches = document["batches"]
    if not isinstance(tasks, dict) or not isinstance(batches, dict):
        raise TaskStoreCorrupt()
    seen_ids = set()
    seen_keys = set()
    batch_members = {}
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
        if task["batch_id"] is not None:
            batch_members.setdefault(task["batch_id"], []).append(task)
    for batch_id, batch in batches.items():
        _validate_batch_summary(batch)
        if batch_id != batch["batch_id"]:
            raise TaskStoreCorrupt()
        members = batch_members.get(batch_id)
        if not members:
            raise TaskStoreCorrupt()
        # Members carry creation-order sequence numbers 0..n-1 with no gaps.
        sequences = sorted(member["sequence"] for member in members)
        if sequences != list(range(len(members))):
            raise TaskStoreCorrupt()
        if batch["sequence"] != len(members):
            raise TaskStoreCorrupt()
    # Every batch reference on a task must resolve to a stored batch.
    if set(batch_members) != set(batches):
        raise TaskStoreCorrupt()


def _empty_document():
    return {"format_version": FORMAT_VERSION, "kind": STORE_KIND,
            "tasks": {}, "batches": {}}


def _load_document(store_path, *, recover=False):
    """Load and validate the store, optionally converging an interrupted publish.

    When ``recover`` is true and the on-disk document is structurally valid but
    contains a ``running`` task journaled with a staged credential publish, the
    publish is completed (or cleaned up) and the document rewritten before it is
    handed out: see ``_recover_pending``. Recovery never rewrites a document
    that is corrupt or illegal; the original bytes are preserved in that case.
    """
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
    if recover:
        try:
            document = _recover_pending(store_path, document)
        except TaskStoreCorrupt:
            raise
        except OSError as error:
            raise TaskStoreIo() from error
    return document


def _fsync_directory(directory):
    """Best-effort fsync of a directory so a rename is durable across crashes."""
    try:
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


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
        _fsync_directory(store_path.parent)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def _pending_staging_path(credential_path):
    return Path(str(credential_path) + _CRED_STAGING_SUFFIX)


def _recover_pending(store_path, document):
    """Converge interrupted attempts left by a crashed or killed process.

    For every ``running`` task the recovery tries to take its per-attempt
    flock; it only proceeds when the lock is free, i.e. the owning process has
    exited (a live executor — possibly another thread in this process sitting
    before the journal — keeps the lock and the task is left untouched).

    Two crash shapes are converged:

    * **journaled publish** (``pending_credential`` set): the credential was
      fully staged and its digest journaled. Intact staged file and no final
      credential -> publish and record ``succeeded``; final credential already
      holding the journaled digest -> record ``succeeded``; otherwise discard
      the staged file and record a retryable ``interrupted`` failure. The
      existing final credential is never overwritten.
    * **claim but no journal**: the owner died before journaling (or before
      proving finished). No new credential can be trusted into place, so any
      stale staging file is removed and the attempt is recorded as a retryable
      ``interrupted`` failure, making the task re-runnable.

    All resolved tasks are committed in one atomic store rewrite. Recovery
    only ever touches the journaled/deterministic staging file, the journaled
    target and that attempt's own lock file.
    """
    running_tasks = [task for task in document["tasks"].values()
                     if task["state"] == STATE_RUNNING]
    if not running_tasks:
        return document
    changed = False
    for task in running_tasks:
        target = Path(task["request"]["credential_path"])
        if _attempt_is_live_locally(store_path, task["task_id"], task["attempt"]):
            # A sibling thread in this process is the live owner (file locks
            # are shared across its threads); leave the attempt untouched.
            continue
        try:
            fence = _acquire_attempt_lock(
                store_path, task["task_id"], task["attempt"])
        except TaskConflict:
            # Still owned by a live executor in another process; leave it.
            continue
        changed = True
        try:
            timestamp = _utcnow()
            journal = task["pending_credential"]
            if journal is None:
                # Owner died before journaling: nothing staged is safe to
                # publish. Reap any staging file and fail retryably.
                with contextlib.suppress(OSError):
                    stale = _pending_staging_path(target)
                    if stale.is_file():
                        stale.unlink()
                _record_terminal(
                    task, {"ok": False, "code": CODE_INTERRUPTED,
                           "retryable": RETRYABLE[CODE_INTERRUPTED]}, timestamp)
                continue

            staged = Path(journal["temp_path"])
            expected_digest = journal["credential_sha256"]

            def staged_is_intact():
                try:
                    return staged.is_file() and _sha256(staged) == expected_digest
                except OSError:
                    return False

            def target_holds_credential():
                try:
                    return target.is_file() and _sha256(target) == expected_digest
                except OSError:
                    return False

            completed = False
            if target_holds_credential():
                completed = True
            elif staged_is_intact():
                try:
                    if not target.exists():
                        os.replace(staged, target)
                        _fsync_directory(target.parent)
                        completed = target_holds_credential()
                except OSError:
                    completed = False
            if not completed:
                # The new credential never made it safely into place: remove
                # only our own deterministic staging file and fall back to a
                # retryable failure.
                with contextlib.suppress(OSError):
                    if staged.resolve() != target.resolve():
                        staged.unlink()
                _record_terminal(
                    task, {"ok": False, "code": CODE_INTERRUPTED,
                           "retryable": RETRYABLE[CODE_INTERRUPTED]}, timestamp)
                continue

            # Publish committed: finish the attempt as succeeded and clear the
            # journal in the same write that publishes the recovered document.
            with contextlib.suppress(OSError):
                if staged.is_file() and staged.resolve() != target.resolve():
                    staged.unlink()
            _record_terminal(task, {"ok": True}, timestamp,
                             credential_digest=expected_digest)
        finally:
            _release_attempt_lock(
                fence, store_path, task["task_id"], task["attempt"])

    if changed:
        # Republish via the normal validated atomic path; any I/O failure here
        # is surfaced (store_io) so the caller never acts on an unpersisted
        # recovery, and the previous file is left intact on a validation error.
        _atomic_write(store_path, document)
    return document


def _record_terminal(task, outcome, timestamp, *, credential_digest=None):
    """Apply a terminal attempt outcome to ``task`` in memory.

    Shared by the normal finish path and crash recovery. The caller guarantees
    the task is ``running`` and that its last attempt is the open one.
    """
    attempt_number = task["attempt"]
    entry = task["attempts"][attempt_number - 1]
    entry["ended_at"] = timestamp
    task["ended_at"] = timestamp
    task["updated_at"] = timestamp
    task["pending_credential"] = None
    if outcome["ok"]:
        digest = credential_digest
        if digest is None:
            raise TaskStoreIo()
        entry["state"] = STATE_SUCCEEDED
        entry["credential_sha256"] = digest
        task["state"] = STATE_SUCCEEDED
        task["credential_sha256"] = digest
        return
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


@contextlib.contextmanager
def _locked_store(store_path, *, create_parents=True, recover=True):
    """Yield ``(document, persist)`` with an exclusive lock on the store.

    Reads happen under the lock so a worker always observes the last committed
    transition. ``persist`` re-validates and atomically replaces the file; on
    any error the previous file is left byte-for-byte intact.

    Read/claim operations pass ``create_parents=False``: a store that does not
    exist at all is presented as empty without creating files or a lock, and
    attempting to persist such a view fails. This keeps ``status``/``run`` on
    a missing store a plain "task not found" rather than an I/O error.

    ``recover`` converges journaled credential publishes from a previous
    process. It is disabled for the live attempt's own finalise transaction:
    that transaction is the one that clears the journal it wrote, and letting
    recovery preempt it would surface a spurious ``conflict``.
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
        # Recovery is safe to run on every open: a journaled publish whose
        # owning attempt still holds its (per-open-fd) flock is skipped, so a
        # live executor — including another thread in this process — is never
        # preempted; only a task whose owner process is dead is converged.
        document = _load_document(
            store_path, recover=recover) if store_path.exists() else _empty_document()

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

    Every filesystem argument is first resolved to an absolute, canonical path
    (symlinks and ``..`` components collapsed) relative to the *current* working
    directory of the ``create`` call. The sealed records therefore never drift
    when a later ``run``/``retry`` is executed from a different directory: the
    worker reopens exactly the files create validated, not files reached by the
    same relative spelling elsewhere.

    Returns the request record to persist. Feature values are validated but
    returned separately so the caller can discard them; they are never stored.
    """
    # Resolve before validation so the pre-flight checks and the sealed paths
    # describe the same canonical files. strict=False keeps the spelling of a
    # not-yet-existing credential target while still absolutising it; its parent
    # is required to exist by _prepare_prove.
    input_path = Path(input_path).resolve()
    model = Path(model).resolve()
    setup_dir = Path(setup_dir).resolve()
    credential_path = Path(credential_path).resolve()
    if not str(input_path) or not str(model) or not str(setup_dir) \
            or not str(credential_path):
        raise TaskError(CODE_INVALID_REQUEST)
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


def _count_queued(document):
    return sum(1 for task in document["tasks"].values()
               if task["state"] == STATE_QUEUED)


def _new_task_record(task_id, request, timestamp, *, idempotency_key,
                     batch_id, sequence):
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
        "pending_credential": None,
        "batch_id": batch_id,
        "sequence": sequence,
    }


def create_task(store_path, input_path, model, setup_dir, credential_path,
                idempotency_key=None, *, max_queued=None):
    """Atomically create a task in ``queued``; honour an idempotency key.

    A repeated create with the same key and a byte-identical request returns
    the original task. The same key with a different request is rejected
    without touching the store. ``max_queued``, when given, bounds the number
    of ``queued`` tasks across the whole store; reaching it is a non-retryable
    ``capacity_exceeded`` and nothing is written.
    """
    if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not 1 <= len(idempotency_key) <= 200):
        raise TaskError(CODE_INVALID_REQUEST)
    if max_queued is not None and (
            not isinstance(max_queued, int) or isinstance(max_queued, bool)
            or max_queued < 0):
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
        if max_queued is not None and _count_queued(document) >= max_queued:
            raise TaskCapacityExceeded()
        task_id = "pt-" + uuid.uuid4().hex[:24]
        while task_id in document["tasks"]:  # pragma: no cover - uuid4 collisions
            task_id = "pt-" + uuid.uuid4().hex[:24]
        task = _new_task_record(
            task_id, request, timestamp, idempotency_key=idempotency_key,
            batch_id=None, sequence=0)
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


def _claim(document, task_id, store_path):
    """Transition one queued task to running and take its attempt fence.

    Runs while the caller holds the store lock. The per-attempt flock is
    acquired here — atomically with the claim transition — so there is no
    window in which the task is running but unfenced: a crash after the claim
    is persisted is always recoverable because the new owner holds the fence
    until finalise. Returns ``(task, attempt_number, fence_handle)``; the
    caller keeps the handle open for the attempt's whole lifetime and releases
    it via ``_release_attempt_lock`` after the terminal state is persisted.
    """
    task = _get_task(document, task_id)
    if task["state"] == STATE_RUNNING:
        raise TaskConflict()
    if task["state"] != STATE_QUEUED:
        raise TaskInvalidState()
    # A queued task has no live executor, so any staging file an earlier
    # attempt left behind is stale and must not survive into this attempt.
    with contextlib.suppress(OSError):
        stale = _pending_staging_path(Path(task["request"]["credential_path"]))
        if stale.is_file():
            stale.unlink()
    timestamp = _utcnow()
    attempt_number = task["attempt"] + 1
    # Acquire the fence before recording the transition: if the (fresh,
    # attempt-numbered) lock cannot be taken the claim is never committed.
    fence = _acquire_attempt_lock(store_path, task_id, attempt_number)
    task["state"] = STATE_RUNNING
    task["attempt"] = attempt_number
    task["started_at"] = timestamp
    task["ended_at"] = None
    task["updated_at"] = timestamp
    task["credential_sha256"] = None
    task["last_error"] = None
    task["pending_credential"] = None
    task["attempts"].append({
        "attempt": attempt_number,
        "state": STATE_RUNNING,
        "started_at": timestamp,
        "ended_at": None,
        "retryable": None,
        "error_code": None,
        "credential_sha256": None,
    })
    return task, attempt_number, fence


def _finish(document, task_id, attempt_number, outcome, *, credential_digest=None):
    """Commit a terminal attempt state, but only if we still own the task."""
    task = _get_task(document, task_id)
    if task["state"] != STATE_RUNNING or task["attempt"] != attempt_number:
        raise TaskConflict()
    entry = task["attempts"][attempt_number - 1]
    if entry["state"] != STATE_RUNNING or entry["attempt"] != attempt_number:
        raise TaskStoreCorrupt()
    _record_terminal(task, outcome, _utcnow(), credential_digest=credential_digest)
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


def _produce_credential(request, staging_path):
    """Run the real EZKL loop and publish only to a sibling staging file.

    Never writes directly to the final credential path: the staged file is
    later renamed into place under the task lock as part of a journaled,
    crash-recoverable commit. A pre-existing file at either the staging or the
    final path is never overwritten by the proving stage.
    """
    target = Path(request["credential_path"])
    if target.exists():
        # The durable contract: success must never clobber a credential that is
        # already there. Fail before any expensive proving.
        raise ZkOutputError("credential already exists")
    staging = Path(staging_path)
    if staging.exists():
        raise ZkOutputError("credential staging file already exists")
    features, paths, model_sha, scale, _credential_path = _prepare_prove(
        request["input_path"], request["model_path"],
        request["setup_dir"], request["credential_path"])
    _issue_credential(features, paths, model_sha, scale, staging)
    return _sha256(staging)


def _execute_attempt(request):
    """Verify materials, run the proof into staging, and return the outcome.

    Returns ``(ok, info)`` where info carries the staged digest on success or
    the failure code/retryable otherwise. Any exception — including
    ``KeyboardInterrupt``/``SystemExit`` and an OSError from the journaling
    window — is converted rather than escaping, so the attempt is always
    finalised by the caller.
    """
    try:
        _verify_request_materials(request)
        staging_path = _pending_staging_path(Path(request["credential_path"]))
        digest = _produce_credential(request, staging_path)
        return True, {"digest": digest, "staging_path": str(staging_path)}
    except BaseException as error:  # noqa: BLE001 - includes KeyboardInterrupt
        if isinstance(error, TaskError):
            raise
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            code, retryable = CODE_INTERRUPTED, RETRYABLE[CODE_INTERRUPTED]
        elif isinstance(error, OSError):
            code, retryable = CODE_OUTPUT_IO, RETRYABLE[CODE_OUTPUT_IO]
        else:
            code, retryable = _classify(error)
        return False, {"code": code, "retryable": retryable}


def _publish_and_finish(store_path, task_id, attempt_number, request):
    """Execute a proof for an already-claimed attempt and journal its publish.

    This is the body shared by single ``run_task`` and the bounded-concurrency
    ``batch_run``: the caller has already committed the ``running`` claim. It
    proves into a staging file, journals the staged publish, renames it into
    place, and commits ``succeeded``/``failed`` — crash-recoverably and
    without ever overwriting an existing final credential.
    """
    ok, info = _execute_attempt(request)
    if ok:
        # Journal the staged publish before the rename: after this commit a
        # crash anywhere in the publish/finalise windows is recoverable.
        journal = {"temp_path": info["staging_path"],
                   "credential_sha256": info["digest"]}
        try:
            with _locked_store(store_path, create_parents=False,
                               recover=False) as (document, persist):
                owner = _get_task(document, task_id)
                if owner["state"] != STATE_RUNNING \
                        or owner["attempt"] != attempt_number:
                    raise TaskConflict()
                owner["pending_credential"] = journal
                persist()
        except TaskError:
            # Journaling failed: discard the staged credential and fail the
            # attempt through the normal finish path below.
            with contextlib.suppress(OSError):
                Path(info["staging_path"]).unlink()
            ok, info = False, {"code": CODE_STORE_IO, "retryable": True}

    if ok:
        target = Path(request["credential_path"])
        staged = Path(info["staging_path"])
        renamed = False
        try:
            if not target.exists():
                os.replace(staged, target)
                _fsync_directory(target.parent)
                renamed = True
        except OSError:
            renamed = False
        if renamed:
            outcome = {"ok": True}
            finish_digest = info["digest"]
        else:
            # Rename impossible/failed, or the target raced into existence:
            # never overwrite it. Drop our staged file and record a retryable
            # output failure; recovery rules keep the final credential safe.
            with contextlib.suppress(OSError):
                if staged.is_file() and staged.resolve() != target.resolve():
                    staged.unlink()
            outcome = {"ok": False, "code": CODE_OUTPUT_IO,
                       "retryable": RETRYABLE[CODE_OUTPUT_IO]}
            finish_digest = None
    else:
        # A failure before the rename leaves no new final credential; remove
        # any staging file the proving stage may have left.
        with contextlib.suppress(OSError):
            staged_path = _pending_staging_path(Path(request["credential_path"]))
            if staged_path.is_file():
                staged_path.unlink()
        outcome = {"ok": False, "code": info["code"], "retryable": info["retryable"]}
        finish_digest = None

    # Finalise under the lock from a fresh on-disk read. The pending journal
    # (if any) is cleared in the same validated write that records the
    # terminal state; recovery is deliberately not re-run here (this is the
    # live attempt that owns the journal). A failure of that bookkeeping
    # (e.g. corrupt store) is a TaskError that must not be masked by the
    # proof outcome.
    with _locked_store(store_path, create_parents=False,
                       recover=False) as (document, persist):
        finished = _finish(document, task_id, attempt_number, outcome,
                           credential_digest=finish_digest)
        persist()
        view = _safe_view(finished)
    if not outcome["ok"]:
        raise TaskAttemptFailed(outcome["code"], outcome["retryable"], view)
    return view


def run_task(store_path, task_id):
    """Claim a queued task, run the real EZKL proof, and record the outcome.

    Only ``queued -> running -> succeeded/failed`` is possible. The claim is
    committed before proving starts, so competing runners fail with
    ``conflict``; an additional per-attempt flock fences the executor for the
    whole attempt so crash recovery never preempts a live worker.

    Credential publication is crash-recoverable: the proof is written to a
    sibling staging file, its digest journaled on the running task in one
    atomic store write, and only then renamed into place; the ``succeeded``
    transition and journal clear land together in the next atomic write. A
    crash in any window is converged on the next store open
    (``_recover_pending``) without ever overwriting an existing final
    credential.
    """
    # Claim and take the attempt fence in one store transaction, then prove
    # without holding the store lock.
    with _locked_store(store_path, create_parents=False) as (document, persist):
        task, attempt_number, fence = _claim(document, task_id, store_path)
        request = task["request"]
        try:
            persist()
        except BaseException:
            # The claim was never durably committed; drop the fence so the lock
            # file does not leak, then surface the (store) failure.
            _release_attempt_lock(fence, store_path, task_id, attempt_number)
            raise
    try:
        return _publish_and_finish(store_path, task_id, attempt_number, request)
    finally:
        _release_attempt_lock(fence, store_path, task_id, attempt_number)


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
        task["pending_credential"] = None
        # last_error is retained as history on the failed attempt entry; the
        # task-level pointer is cleared for the fresh execution.
        task["last_error"] = None
        persist()
        return _safe_view(task)


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------

_ITEM_ARG_KEYS = ("input", "model", "setup_dir", "credential")


def _coerce_item(raw, index):
    """Validate one batch-create item's plain JSON shape.

    The path strings themselves are validated like a single ``create`` by
    ``_build_request`` during the all-or-nothing preflight. ``index`` keeps the
    signature stable for ordered diagnostics.
    """
    del index
    if not isinstance(raw, dict):
        raise TaskError(CODE_INVALID_REQUEST)
    item = {}
    for key in _ITEM_ARG_KEYS:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise TaskError(CODE_INVALID_REQUEST)
        item[key] = value
    allowed = set(_ITEM_ARG_KEYS) | {"idempotency_key"}
    if set(raw) - allowed:
        raise TaskError(CODE_INVALID_REQUEST)
    key = raw.get("idempotency_key")
    if key is not None and (not isinstance(key, str) or not 1 <= len(key) <= 200):
        raise TaskError(CODE_INVALID_REQUEST)
    item["idempotency_key"] = key
    return item


def _get_batch(document, batch_id):
    if not isinstance(batch_id, str) or not _BATCH_ID_RE.fullmatch(batch_id):
        raise TaskError(CODE_INVALID_REQUEST)
    batch = document["batches"].get(batch_id)
    if batch is None:
        raise TaskBatchNotFound()
    return batch


def _batch_views(document, batch_id):
    members = [task for task in document["tasks"].values()
               if task["batch_id"] == batch_id]
    members.sort(key=lambda task: task["sequence"])
    return [_safe_view(task) for task in members]


def _batch_payload(document, batch_id):
    """Total, the four-state counts and creation-order safe task views."""
    _get_batch(document, batch_id)
    views = _batch_views(document, batch_id)
    counts = {state: 0 for state in BATCH_STATES}
    for view in views:
        counts[view["state"]] += 1
    return {"batch_id": batch_id, "total": len(views),
            "counts": counts, "tasks": views}


def batch_create(store_path, raw_items, *, max_queued=None):
    """Create a batch of queued tasks all-or-nothing; return its safe view.

    Every item is pre-flighted exactly like a single ``create`` (features,
    manifest, model and all setup artifacts) before the batch id exists; if any
    item is invalid or collides on an idempotency key, the whole batch is
    rejected and the store is never written. On success the batch summary and
    every member task land in one atomic store update.

    Replaying a batch-create whose items all carry idempotency keys that
    resolve to members of one existing batch returns that same batch
    (``created=False``) with a stable ``batch_id``. ``max_queued`` bounds the
    number of queued tasks across the whole store; exceeding it is a
    non-retryable ``capacity_exceeded`` and nothing is written.
    """
    if max_queued is not None and (
            not isinstance(max_queued, int) or isinstance(max_queued, bool)
            or max_queued < 0):
        raise TaskError(CODE_INVALID_REQUEST)
    if not isinstance(raw_items, list) or not raw_items:
        raise TaskError(CODE_INVALID_REQUEST)
    items = [_coerce_item(raw, index) for index, raw in enumerate(raw_items)]
    # A key repeated inside one file cannot identify two distinct tasks.
    file_keys = [item["idempotency_key"] for item in items
                 if item["idempotency_key"] is not None]
    if len(set(file_keys)) != len(file_keys):
        raise TaskIdempotencyConflict()

    # All-or-nothing preflight, before the store (and the batch id) exists.
    prepared = []
    try:
        for item in items:
            request = _build_request(
                item["input"], item["model"], item["setup_dir"], item["credential"])
            prepared.append((item, request))
    except TaskError:
        raise
    except ZkError as error:
        code, retryable = _classify(error)
        raise TaskError(code, retryable=retryable) from None

    timestamp = _utcnow()
    with _locked_store(store_path) as (document, persist):
        existing_by_key = {}
        for task in document["tasks"].values():
            if task["idempotency_key"] is not None:
                existing_by_key[task["idempotency_key"]] = task

        resolved = []
        for item, request in prepared:
            key = item["idempotency_key"]
            if key is None:
                resolved.append((item, request, None))
                continue
            existing = existing_by_key.get(key)
            if existing is None:
                resolved.append((item, request, None))
            elif existing["request"]["request_sha256"] == request["request_sha256"]:
                resolved.append((item, request, existing))
            else:
                raise TaskIdempotencyConflict()

        reused = [task for _item, _request, task in resolved if task is not None]
        if len(reused) == len(resolved):
            # Full idempotent replay: every item maps to an existing task. They
            # must all be members of one batch, whose stable id is returned.
            batch_ids = {task["batch_id"] for task in reused}
            if len(batch_ids) == 1 and batch_ids.pop() in document["batches"]:
                replay_id = reused[0]["batch_id"]
                return _batch_payload(document, replay_id), False
            # Keys resolve to standalone tasks or to mixed batches: a new batch
            # cannot be formed around them, and batches never absorb members.
            raise TaskIdempotencyConflict()
        if reused:
            # Partial overlap would mix new and existing members: reject
            # without writing so batches stay immutable all-or-nothing groups.
            raise TaskIdempotencyConflict()

        new_count = len(resolved)
        if max_queued is not None and _count_queued(document) + new_count > max_queued:
            raise TaskCapacityExceeded()

        batch_id = "pb-" + uuid.uuid4().hex[:24]
        while batch_id in document["batches"]:  # pragma: no cover - uuid4 collisions
            batch_id = "pb-" + uuid.uuid4().hex[:24]
        for sequence, (_item, request, _existing) in enumerate(resolved):
            task_id = "pt-" + uuid.uuid4().hex[:24]
            while task_id in document["tasks"]:  # pragma: no cover
                task_id = "pt-" + uuid.uuid4().hex[:24]
            task = _new_task_record(
                task_id, request, timestamp,
                idempotency_key=_item["idempotency_key"],
                batch_id=batch_id, sequence=sequence)
            document["tasks"][task_id] = task
        document["batches"][batch_id] = {
            "batch_id": batch_id,
            "created_at": timestamp,
            "sequence": new_count,
        }
        persist()
        return _batch_payload(document, batch_id), True


def batch_status(store_path, batch_id):
    """Return total, four-state counts and ordered safe views for one batch."""
    with _locked_store(store_path, create_parents=False) as (document, _persist):
        return _batch_payload(document, batch_id)


def _claim_next_batch_task(document, batch_id, store_path):
    """Atomically claim the batch's next queued task (and its fence)."""
    _get_batch(document, batch_id)
    candidates = [task for task in document["tasks"].values()
                  if task["batch_id"] == batch_id and task["state"] == STATE_QUEUED]
    if not candidates:
        return None
    candidates.sort(key=lambda task: task["sequence"])
    task, attempt_number, fence = _claim(
        document, candidates[0]["task_id"], store_path)
    return task["task_id"], attempt_number, task["request"], fence


def batch_run(store_path, batch_id, max_workers):
    """Run a batch's queued tasks with bounded concurrency.

    Only tasks belonging to this batch are ever claimed, and only under the
    exclusive store lock, so a concurrent single ``run``/another batch can
    never take the same task. At most ``max_workers`` proofs execute at once;
    one item failing never blocks or cancels the others. Each task keeps the
    ordinary single-task claim/attempt/retry semantics and performs the real
    EZKL proof. Returns the post-run batch payload; per-task failures are
    reflected in the state counts, not raised.
    """
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) \
            or max_workers < 1:
        raise TaskError(CODE_INVALID_REQUEST)

    # Validate the batch exists before spawning any worker.
    with _locked_store(store_path, create_parents=False) as (document, _persist):
        _get_batch(document, batch_id)

    def claim_next():
        """Claim one batch task and its fence, durably, under the store lock."""
        with _locked_store(store_path, create_parents=False) as (document, persist):
            claimed = _claim_next_batch_task(document, batch_id, store_path)
            if claimed is not None:
                try:
                    persist()
                except BaseException:
                    _release_attempt_lock(
                        claimed[3], store_path, claimed[0], claimed[1])
                    raise
            return claimed

    def worker():
        while True:
            try:
                claimed = claim_next()
            except TaskError:
                # A store-level control/corruption error on claim: stop this
                # worker; remaining tasks stay queued/running for a later run.
                return
            if claimed is None:
                return
            task_id, attempt_number, request, fence = claimed
            try:
                _publish_and_finish(store_path, task_id, attempt_number, request)
            except TaskAttemptFailed:
                # Recorded on the task; the batch result reports it. Keep
                # claiming the rest of the batch.
                continue
            except TaskError:
                # The attempt could not be finalised (e.g. the store became
                # unreadable): its fence is released below, so the task is
                # reconciled by recovery on the next open. Stop this worker;
                # other workers drain whatever they can still claim.
                return
            except Exception:  # noqa: BLE001 - never leak a traceback from a worker
                # An unexpected in-worker failure leaves the task running with
                # the fence released; recovery converges it. Other items are
                # not blocked.
                continue
            finally:
                _release_attempt_lock(fence, store_path, task_id, attempt_number)

    threads = [threading.Thread(target=worker, name=f"batch-{batch_id}-{i}")
               for i in range(max_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with _locked_store(store_path, create_parents=False) as (document, _persist):
        return _batch_payload(document, batch_id)
