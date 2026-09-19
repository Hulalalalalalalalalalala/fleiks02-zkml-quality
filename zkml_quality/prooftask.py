"""Offline prover-side task queue for the EZKL proof loop.

``proof-task`` adds a small durable task library (``--store``) on top of the
existing ``zk-prove`` loop so proving work can be created, executed, inspected
and retried as discrete units:

* ``create`` validates the same inputs as ``zk-prove`` (the private feature
  input, the ONNX model, the ``zk-setup`` directory and the credential
  target) and atomically records a new task in the ``queued`` state. An
  optional ``--idempotency-key`` makes a repeated identical submission return
  the original task; the same key with a different request is a conflict and
  is rejected.
* ``run`` performs the only executor transition: ``queued`` -> ``running``
  -> ``succeeded``/``failed``. Each run increments ``attempt`` and records
  the start/finish times; a per-task executor lock guarantees a single
  executor, so a competing ``run`` fails. Success always means a real EZKL
  proof was generated and self-verified; only then is the credential
  SHA-256 recorded. A failure or interruption never reports success, never
  overwrites a credential and never drops attempt history.
* ``status`` reports the current task without modifying the store.
* ``retry`` re-queues a ``failed`` task whose error is retryable; every
  other state is refused.

The store is a directory of per-task JSON records plus an idempotency index.
Every record is validated strictly on load; a corrupt store or an illegal
state is rejected and the original file is left byte-for-byte intact. All
updates go through a same-directory temporary file and ``os.replace``.

Privacy: command output carries only the task id, state, attempt, timestamps,
the credential digest and, for failed tasks, a sanitised error. It never
contains features, input content, proof data or filesystem paths; the paths a
task needs stay inside the prover-owned store and are never printed.
"""
import contextlib
import hashlib
import json
import os
import re
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .inference import validate_features
from .zk import ARTIFACT_NAMES, MANIFEST_NAME, ZkError, _read_manifest, _sha256, run_prove

FORMAT_VERSION = 1
TASK_KIND = "proof-task"
IDEM_KIND = "proof-task-idempotency"

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_SUCCEEDED = "succeeded"
STATE_FAILED = "failed"
STATES = (STATE_QUEUED, STATE_RUNNING, STATE_SUCCEEDED, STATE_FAILED)

ERR_INVALID_REQUEST = "invalid_request"
ERR_ARTIFACT_MISSING = "artifact_missing"
ERR_DIGEST_MISMATCH = "digest_mismatch"
ERR_EZKL_FAILURE = "ezkl_failure"
ERR_OUTPUT_IO = "output_io"
ERR_INTERRUPTED = "interrupted"
ERR_STORE_CORRUPT = "store_corrupt"
# Codes that may be recorded on a failed task. ``store_corrupt`` is only ever
# a command rejection: a corrupt store is refused, never rewritten.
RECORD_ERROR_CODES = (
    ERR_INVALID_REQUEST,
    ERR_ARTIFACT_MISSING,
    ERR_DIGEST_MISMATCH,
    ERR_EZKL_FAILURE,
    ERR_OUTPUT_IO,
    ERR_INTERRUPTED,
)

TASK_ID_RE = re.compile(r"pt-[0-9a-f]{32}")
MAX_IDEMPOTENCY_KEY_LENGTH = 256

_HEX64 = set("0123456789abcdef")


class ProofTaskError(Exception):
    """Raised for any rejected, failed or interrupted proof-task operation.

    Carries a stable machine-readable ``code``, a sanitised ``message`` that
    never contains paths, features or proof data, and whether a failed task
    may be re-queued by ``retry``.
    """

    def __init__(self, code, message, retryable=False):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _is_sha256_hex(value):
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX64 for char in value)


def _is_timestamp(value):
    if not isinstance(value, str) or not value:
        return False
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ProofTaskError(ERR_STORE_CORRUPT, "task store contains a duplicate key")
        document[key] = value
    return document


def _load_json_file(path, not_found_error):
    path = Path(path)
    if not path.is_file():
        raise not_found_error
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_reject_duplicate_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ProofTaskError(ERR_STORE_CORRUPT,
                                             "task store contains an invalid JSON constant")))
    except ProofTaskError:
        raise
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ProofTaskError(ERR_STORE_CORRUPT,
                             "task store is corrupt or contains an illegal state") from error


def _validate_error_object(error):
    return (isinstance(error, dict)
            and set(error) == {"code", "retryable", "message"}
            and error["code"] in RECORD_ERROR_CODES
            and isinstance(error["retryable"], bool)
            and isinstance(error["message"], str) and bool(error["message"]))


_REQUEST_PATH_FIELDS = ("input_path", "model_path", "setup_dir", "credential_path")
_REQUEST_DIGEST_FIELDS = ("input_sha256", "model_sha256", "manifest_sha256")


def _validate_task_record(record, task_id):
    """Strictly validate a stored task record, including state invariants."""
    valid = isinstance(record, dict) and set(record) == {
        "format_version", "kind", "task_id", "state", "attempt",
        "created_at", "started_at", "finished_at",
        "request", "credential_sha256", "error", "history",
    }
    if valid:
        valid = (record["format_version"] == FORMAT_VERSION
                 and record["kind"] == TASK_KIND
                 and record["task_id"] == task_id
                 and record["state"] in STATES
                 and isinstance(record["attempt"], int)
                 and not isinstance(record["attempt"], bool)
                 and record["attempt"] >= 0
                 and _is_timestamp(record["created_at"])
                 and (record["started_at"] is None or _is_timestamp(record["started_at"]))
                 and (record["finished_at"] is None or _is_timestamp(record["finished_at"]))
                 and (record["credential_sha256"] is None
                      or _is_sha256_hex(record["credential_sha256"]))
                 and (record["error"] is None or _validate_error_object(record["error"])))
    if valid:
        request = record["request"]
        valid = (isinstance(request, dict)
                 and set(request) == set(_REQUEST_PATH_FIELDS) | set(_REQUEST_DIGEST_FIELDS)
                 and all(isinstance(request[field], str) and request[field]
                         for field in _REQUEST_PATH_FIELDS)
                 and all(_is_sha256_hex(request[field]) for field in _REQUEST_DIGEST_FIELDS))
    if valid:
        history = record["history"]
        valid = isinstance(history, list) and all(
            isinstance(entry, dict)
            and set(entry) == {"attempt", "started_at", "finished_at", "result", "error"}
            and isinstance(entry["attempt"], int) and not isinstance(entry["attempt"], bool)
            and entry["attempt"] >= 1
            and _is_timestamp(entry["started_at"]) and _is_timestamp(entry["finished_at"])
            and entry["result"] in (STATE_SUCCEEDED, STATE_FAILED)
            and (entry["error"] is None if entry["result"] == STATE_SUCCEEDED
                 else _validate_error_object(entry["error"]))
            for entry in history)
    if valid:
        # State-machine invariants: anything else is an illegal state.
        state = record["state"]
        if state == STATE_QUEUED:
            valid = (record["started_at"] is None and record["finished_at"] is None
                     and record["error"] is None and record["credential_sha256"] is None)
        elif state == STATE_RUNNING:
            valid = (record["attempt"] >= 1 and record["started_at"] is not None
                     and record["finished_at"] is None and record["error"] is None
                     and record["credential_sha256"] is None)
        elif state == STATE_SUCCEEDED:
            valid = (record["attempt"] >= 1 and record["started_at"] is not None
                     and record["finished_at"] is not None and record["error"] is None
                     and record["credential_sha256"] is not None)
        else:  # failed
            valid = (record["attempt"] >= 1 and record["started_at"] is not None
                     and record["finished_at"] is not None and record["error"] is not None
                     and record["credential_sha256"] is None)
    if not valid:
        raise ProofTaskError(ERR_STORE_CORRUPT,
                             "task store is corrupt or contains an illegal state")


def _validate_idem_record(record):
    valid = isinstance(record, dict) and set(record) == {
        "format_version", "kind", "key_sha256", "request_sha256", "task_id", "created_at",
    }
    if valid:
        valid = (record["format_version"] == FORMAT_VERSION
                 and record["kind"] == IDEM_KIND
                 and _is_sha256_hex(record["key_sha256"])
                 and _is_sha256_hex(record["request_sha256"])
                 and TASK_ID_RE.fullmatch(record["task_id"]) is not None
                 and _is_timestamp(record["created_at"]))
    if not valid:
        raise ProofTaskError(ERR_STORE_CORRUPT,
                             "task store is corrupt or contains an illegal state")


def _atomic_write_json(path, record):
    """Publish a record via a same-directory temp file and ``os.replace``."""
    path = Path(path)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".task-", suffix=".tmp", delete=False)
    try:
        json.dump(record, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            os.unlink(handle.name)
        raise


def _unlink_quietly(path):
    with contextlib.suppress(OSError):
        Path(path).unlink()


def _require_store(store):
    store = Path(store)
    if not store.is_dir():
        raise ProofTaskError(ERR_STORE_CORRUPT, "task store not found")
    return store


def _tasks_dir(store):
    return Path(store) / "tasks"


def _task_path(store, task_id):
    return _tasks_dir(store) / f"{task_id}.json"


def _lock_path(store, task_id):
    return Path(store) / "locks" / f"{task_id}.lock"


def _validate_task_id(task_id):
    if not isinstance(task_id, str) or TASK_ID_RE.fullmatch(task_id) is None:
        raise ProofTaskError(ERR_INVALID_REQUEST, "task id is invalid")
    return task_id


def _load_task(store, task_id):
    record = _load_json_file(
        _task_path(store, task_id),
        ProofTaskError(ERR_INVALID_REQUEST, "task not found"))
    _validate_task_record(record, task_id)
    return record


def _output(record):
    """The public view of a task: no paths, features, input or proof data."""
    result = {
        "task_id": record["task_id"],
        "state": record["state"],
        "attempt": record["attempt"],
        "created_at": record["created_at"],
        "started_at": record["started_at"],
        "finished_at": record["finished_at"],
        "credential_sha256": record["credential_sha256"],
    }
    if record["error"] is not None:
        result["error"] = dict(record["error"])
    return result


def _request_fingerprint(request):
    payload = {field: request[field] for field in
               ("credential_path",) + _REQUEST_DIGEST_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_idempotency_key(key):
    if not isinstance(key, str) or not key or len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise ProofTaskError(ERR_INVALID_REQUEST, "idempotency key is invalid")
    return key


def _idem_path(store, key):
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return Path(store) / "idempotency" / f"{digest}.json"


def _load_idem_record(path):
    path = Path(path)
    if not path.is_file():
        return None
    record = _load_json_file(
        path, ProofTaskError(ERR_STORE_CORRUPT,
                             "task store is corrupt or contains an illegal state"))
    _validate_idem_record(record)
    return record


def _validate_request(input_path, model, setup_dir, credential_path):
    """Validate exactly what ``zk-prove`` would need, without proving.

    Everything that is wrong with the request itself is an ``invalid_request``
    rejection at create time; nothing is written to the store.
    """
    input_path = Path(input_path)
    model = Path(model)
    setup_dir = Path(setup_dir)
    credential_path = Path(credential_path)

    if not input_path.is_file():
        raise ProofTaskError(ERR_INVALID_REQUEST, "input file not found")
    try:
        document = json.loads(input_path.read_text(encoding="utf-8"),
                              parse_constant=lambda value: (_ for _ in ()).throw(
                                  ValueError(f"invalid JSON constant {value}")))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ProofTaskError(ERR_INVALID_REQUEST, "input file is not valid JSON") from error
    try:
        validate_features(document)  # identical six-dimension contract as `infer`
    except ValueError as error:
        raise ProofTaskError(ERR_INVALID_REQUEST, str(error)) from error

    if not model.is_file():
        raise ProofTaskError(ERR_INVALID_REQUEST, "ONNX model not found")
    if not setup_dir.is_dir():
        raise ProofTaskError(ERR_INVALID_REQUEST, "setup directory not found")
    manifest_path = setup_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ProofTaskError(ERR_INVALID_REQUEST, "setup manifest not found")
    try:
        manifest = _read_manifest(manifest_path)
    except ZkError as error:
        raise ProofTaskError(ERR_INVALID_REQUEST, "setup manifest is invalid") from error
    for name in ARTIFACT_NAMES.values():
        if not (setup_dir / name).is_file():
            raise ProofTaskError(ERR_INVALID_REQUEST, "setup artifact is missing")
    model_sha = _sha256(model)
    if model_sha != manifest["model_sha256"]:
        raise ProofTaskError(ERR_INVALID_REQUEST,
                             "model does not match the model this setup was built for")
    if credential_path.is_dir():
        raise ProofTaskError(ERR_INVALID_REQUEST, "credential path is a directory")
    if not credential_path.resolve().parent.is_dir():
        raise ProofTaskError(ERR_INVALID_REQUEST,
                             "credential output directory does not exist")

    return {
        "input_path": str(input_path.resolve()),
        "model_path": str(model.resolve()),
        "setup_dir": str(setup_dir.resolve()),
        "credential_path": str(credential_path.resolve()),
        "input_sha256": _sha256(input_path),
        "model_sha256": model_sha,
        "manifest_sha256": _sha256(manifest_path),
    }


def _new_task_record(request):
    return {
        "format_version": FORMAT_VERSION,
        "kind": TASK_KIND,
        "task_id": None,
        "state": STATE_QUEUED,
        "attempt": 0,
        "created_at": None,
        "started_at": None,
        "finished_at": None,
        "request": request,
        "credential_sha256": None,
        "error": None,
        "history": [],
    }


def _create_task_file(tasks_dir, record):
    """Atomically allocate a task id and persist the new queued task."""
    for _ in range(100):
        task_id = f"pt-{secrets.token_hex(16)}"
        path = tasks_dir / f"{task_id}.json"
        record["task_id"] = task_id
        record["created_at"] = _now()
        _validate_task_record(record, task_id)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, allow_nan=False,
                          indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            _unlink_quietly(path)
            raise
        return task_id, path
    raise ProofTaskError(ERR_INVALID_REQUEST, "could not allocate a task id")


def _resolve_idempotent(store, idem_record, fingerprint):
    """Return the original task for an identical request; reject conflicts."""
    if idem_record["request_sha256"] != fingerprint:
        raise ProofTaskError(ERR_INVALID_REQUEST,
                             "idempotency key was already used with a different request")
    record = _load_task(store, idem_record["task_id"])
    return _output(record)


def create_task(store, input_path, model, setup_dir, credential_path, idempotency_key=None):
    """Validate the request and atomically record a new ``queued`` task."""
    request = _validate_request(input_path, model, setup_dir, credential_path)
    fingerprint = _request_fingerprint(request)

    store = Path(store)
    if store.exists() and not store.is_dir():
        raise ProofTaskError(ERR_STORE_CORRUPT, "task store path is not a directory")
    tasks_dir = _tasks_dir(store)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    idem_path = None
    if idempotency_key is not None:
        key = _validate_idempotency_key(idempotency_key)
        idem_path = _idem_path(store, key)
        (store / "idempotency").mkdir(parents=True, exist_ok=True)
        existing = _load_idem_record(idem_path)
        if existing is not None:
            return _resolve_idempotent(store, existing, fingerprint)

    record = _new_task_record(request)
    _, task_path = _create_task_file(tasks_dir, record)

    if idem_path is not None:
        idem_record = {
            "format_version": FORMAT_VERSION,
            "kind": IDEM_KIND,
            "key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "request_sha256": fingerprint,
            "task_id": record["task_id"],
            "created_at": record["created_at"],
        }
        _validate_idem_record(idem_record)
        try:
            fd = os.open(idem_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            # Lost a concurrent create race: discard our task and resolve
            # against the record that won.
            try:
                existing = _load_idem_record(idem_path)
            finally:
                _unlink_quietly(task_path)
            return _resolve_idempotent(store, existing, fingerprint)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(idem_record, handle, ensure_ascii=False, allow_nan=False,
                          indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            _unlink_quietly(idem_path)
            _unlink_quietly(task_path)
            raise
    return _output(record)


def status_task(store, task_id):
    """Return the public view of a task; the store is never modified."""
    store = _require_store(store)
    task_id = _validate_task_id(task_id)
    return _output(_load_task(store, task_id))


def _execute(request):
    """Run the real EZKL proof for a task and return the credential SHA-256.

    Every artifact named by the task must still exist and still hash to the
    digest pinned at create time; the setup artifacts must still match the
    setup manifest. Only a genuine, self-verified EZKL proof counts as
    success.
    """
    input_path = Path(request["input_path"])
    model = Path(request["model_path"])
    setup_dir = Path(request["setup_dir"])
    credential_path = Path(request["credential_path"])
    manifest_path = setup_dir / MANIFEST_NAME

    if not setup_dir.is_dir() or not input_path.is_file() or not model.is_file() \
            or not manifest_path.is_file():
        raise ProofTaskError(ERR_ARTIFACT_MISSING, "a required artifact is missing",
                             retryable=True)
    artifacts = {key: setup_dir / name for key, name in ARTIFACT_NAMES.items()}
    if any(not path.is_file() for path in artifacts.values()):
        raise ProofTaskError(ERR_ARTIFACT_MISSING, "a required artifact is missing",
                             retryable=True)

    if _sha256(input_path) != request["input_sha256"] \
            or _sha256(model) != request["model_sha256"] \
            or _sha256(manifest_path) != request["manifest_sha256"]:
        raise ProofTaskError(ERR_DIGEST_MISMATCH,
                             "artifact content does not match the task record")
    try:
        manifest = _read_manifest(manifest_path)
    except ZkError as error:
        raise ProofTaskError(ERR_EZKL_FAILURE, "setup manifest is not usable",
                             retryable=True) from error
    if any(_sha256(path) != manifest["artifacts"][key] for key, path in artifacts.items()):
        raise ProofTaskError(ERR_DIGEST_MISMATCH,
                             "artifact content does not match the task record")

    if credential_path.is_dir() or not credential_path.parent.is_dir():
        raise ProofTaskError(ERR_OUTPUT_IO, "credential output target is not writable",
                             retryable=True)
    try:
        run_prove(input_path, model, setup_dir, credential_path)
    except ProofTaskError:
        raise
    except ZkError as error:
        raise ProofTaskError(ERR_EZKL_FAILURE, "proof generation failed",
                             retryable=True) from error
    except OSError as error:
        raise ProofTaskError(ERR_OUTPUT_IO, "could not write the credential output",
                             retryable=True) from error
    except Exception as error:
        raise ProofTaskError(ERR_EZKL_FAILURE, "proof generation failed",
                             retryable=True) from error
    if not credential_path.is_file():
        raise ProofTaskError(ERR_EZKL_FAILURE, "proof generation failed", retryable=True)
    try:
        return _sha256(credential_path)
    except OSError as error:
        raise ProofTaskError(ERR_OUTPUT_IO, "could not read the credential output",
                             retryable=True) from error


def _record_failure(task_path, record, error):
    record["state"] = STATE_FAILED
    record["finished_at"] = _now()
    record["error"] = error
    record["history"].append({
        "attempt": record["attempt"],
        "started_at": record["started_at"],
        "finished_at": record["finished_at"],
        "result": STATE_FAILED,
        "error": error,
    })
    _validate_task_record(record, record["task_id"])
    _atomic_write_json(task_path, record)


def run_task(store, task_id):
    """Execute one queued task: ``queued`` -> ``running`` -> terminal state.

    Only a queued task may run and only one executor may hold a task; any
    other call fails without modifying the store. A failed or interrupted
    execution is recorded with a sanitised, classified error and re-raised so
    the command exits nonzero having written nothing to stdout.
    """
    store = _require_store(store)
    task_id = _validate_task_id(task_id)
    task_path = _task_path(store, task_id)
    record = _load_task(store, task_id)
    if record["state"] != STATE_QUEUED:
        raise ProofTaskError(ERR_INVALID_REQUEST,
                             f"task is {record['state']}; only queued tasks can run")

    locks_dir = Path(store) / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(store, task_id)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ProofTaskError(ERR_INVALID_REQUEST,
                             "another executor is already running this task") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid {os.getpid()}\n")

        record["state"] = STATE_RUNNING
        record["attempt"] += 1
        record["started_at"] = _now()
        record["finished_at"] = None
        record["error"] = None
        _validate_task_record(record, task_id)
        _atomic_write_json(task_path, record)

        try:
            credential_sha = _execute(record["request"])
        except KeyboardInterrupt as error:
            failure = {"code": ERR_INTERRUPTED, "retryable": True,
                       "message": "execution was interrupted"}
            _record_failure(task_path, record, failure)
            raise ProofTaskError(ERR_INTERRUPTED, failure["message"],
                                 retryable=True) from error
        except ProofTaskError as error:
            _record_failure(task_path, record, {
                "code": error.code, "retryable": error.retryable, "message": error.message,
            })
            raise

        record["state"] = STATE_SUCCEEDED
        record["finished_at"] = _now()
        record["credential_sha256"] = credential_sha
        record["history"].append({
            "attempt": record["attempt"],
            "started_at": record["started_at"],
            "finished_at": record["finished_at"],
            "result": STATE_SUCCEEDED,
            "error": None,
        })
        _validate_task_record(record, task_id)
        _atomic_write_json(task_path, record)
        return _output(record)
    finally:
        _unlink_quietly(lock_path)


def retry_task(store, task_id):
    """Re-queue a ``failed`` task whose recorded error is retryable."""
    store = _require_store(store)
    task_id = _validate_task_id(task_id)
    task_path = _task_path(store, task_id)
    record = _load_task(store, task_id)
    if record["state"] != STATE_FAILED:
        raise ProofTaskError(ERR_INVALID_REQUEST,
                             f"task is {record['state']}; only failed tasks can be retried")
    if not record["error"]["retryable"]:
        raise ProofTaskError(ERR_INVALID_REQUEST, "task failure is not retryable")
    record["state"] = STATE_QUEUED
    record["error"] = None
    record["started_at"] = None
    record["finished_at"] = None
    _validate_task_record(record, task_id)
    _atomic_write_json(task_path, record)
    return _output(record)
