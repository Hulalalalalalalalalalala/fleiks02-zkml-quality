import argparse
import json
import sys
from pathlib import Path

from .inference import ROOT, infer
from .registry import (
    RegistryError,
    enable_model,
    list_models,
    register_model,
    revoke_model,
)
from .tasks import (
    SAFE_MESSAGES,
    TaskAttemptFailed,
    TaskError,
    create_task,
    retry_task,
    run_task,
    status_task,
)
from .zk import ZkError, run_prove, run_setup, run_verify


def _parser():
    parser = argparse.ArgumentParser(description="Equipment-quality inference example")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("infer", help="Evaluate one JSON feature vector")
    command.add_argument("--input", required=True, type=Path)

    sub.add_parser("demo", help="Evaluate the bundled synthetic samples")

    setup = sub.add_parser("zk-setup", help="Compile the circuit and generate SRS and keys")
    setup.add_argument("--model", required=True, type=Path, help="ONNX model file")
    setup.add_argument("--dir", required=True, type=Path, help="Setup output directory")

    prove = sub.add_parser("zk-prove", help="Prove a private feature vector with EZKL (CPU)")
    prove.add_argument("--input", required=True, type=Path, help="JSON input containing features")
    prove.add_argument("--model", required=True, type=Path, help="ONNX model used at setup")
    prove.add_argument("--setup-dir", required=True, type=Path, help="Directory from zk-setup")
    prove.add_argument("--credential", required=True, type=Path, help="Credential JSON to write")

    verify = sub.add_parser(
        "zk-verify",
        help="Verify a credential against an enabled registry record and verifier-owned manifest; "
             "no input, proving key or network")
    verify.add_argument("--credential", required=True, type=Path)
    verify.add_argument(
        "--registry", required=True, type=Path,
        help="verifier-owned local model registry JSON; only enabled versions are admitted")
    verify.add_argument(
        "--model-version", required=True,
        help="registry version ([A-Za-z0-9._-]+) the credential is verified against")
    verify.add_argument(
        "--manifest", required=True, type=Path,
        help="manifest.json from a zk-setup run the verifier independently obtained and trusts")
    verify.add_argument("--model", required=True, type=Path, help="Verifier-chosen ONNX model")
    verify.add_argument("--settings", required=True, type=Path)
    verify.add_argument("--vk", required=True, type=Path, help="Verification key")
    verify.add_argument("--srs", required=True, type=Path)

    registry = sub.add_parser(
        "model-registry",
        help="Manage the local offline model registry (register/list/enable/revoke)")
    registry_sub = registry.add_subparsers(dest="registry_command", required=True)

    register = registry_sub.add_parser(
        "register", help="Register an approved manifest/model version (starts disabled)")
    register.add_argument("--registry", required=True, type=Path, help="Registry JSON file")
    register.add_argument(
        "--version", required=True,
        help="Model version token matching [A-Za-z0-9._-]+")
    register.add_argument("--manifest", required=True, type=Path, help="Approved setup manifest")
    register.add_argument("--model", required=True, type=Path, help="Approved ONNX model")

    list_cmd = registry_sub.add_parser(
        "list", help="List registered versions in lexicographic order")
    list_cmd.add_argument("--registry", required=True, type=Path, help="Registry JSON file")

    enable = registry_sub.add_parser("enable", help="Enable a disabled version")
    enable.add_argument("--registry", required=True, type=Path, help="Registry JSON file")
    enable.add_argument(
        "--version", required=True,
        help="Model version token matching [A-Za-z0-9._-]+")

    revoke = registry_sub.add_parser("revoke", help="Irreversibly revoke a version")
    revoke.add_argument("--registry", required=True, type=Path, help="Registry JSON file")
    revoke.add_argument(
        "--version", required=True,
        help="Model version token matching [A-Za-z0-9._-]+")

    proof_task = sub.add_parser(
        "proof-task",
        help="Manage durable prover tasks (create/status/run/retry) in a local task store")
    task_sub = proof_task.add_subparsers(dest="task_command", required=True)

    def add_store(target):
        target.add_argument(
            "--store", required=True, type=Path,
            help="prover-owned proof-task store JSON (created atomically if absent)")

    def add_prove_arguments(target):
        target.add_argument("--input", required=True, type=Path,
                            help="JSON input containing features (same contract as zk-prove)")
        target.add_argument("--model", required=True, type=Path,
                            help="ONNX model used at setup")
        target.add_argument("--setup-dir", required=True, type=Path,
                            help="Directory from zk-setup")
        target.add_argument("--credential", required=True, type=Path,
                            help="Credential JSON the task writes when it succeeds")

    task_create = task_sub.add_parser(
        "create", help="Atomically create a queued proof task")
    add_store(task_create)
    add_prove_arguments(task_create)
    task_create.add_argument(
        "--idempotency-key",
        help="if given, repeat the same key with the identical request to get "
             "the original task; a different request with the same key is rejected")

    task_status = task_sub.add_parser("status", help="Show one task's state")
    add_store(task_status)
    task_status.add_argument("--task-id", required=True)

    task_run = task_sub.add_parser(
        "run", help="Claim a queued task and perform the real EZKL proof")
    add_store(task_run)
    task_run.add_argument("--task-id", required=True)

    task_retry = task_sub.add_parser(
        "retry", help="Re-queue a failed retryable task")
    add_store(task_retry)
    task_retry.add_argument("--task-id", required=True)
    return parser


def _print_error(error, view=None):
    """Emit exactly one JSON object describing the failure on stderr.

    A failed ``run`` still reports the post-transition task state: the task
    fields are merged with the ``error`` block. Nothing is ever written to
    stdout for a failed command.
    """
    if isinstance(error, TaskError):
        code = error.code
        retryable = error.retryable
    else:
        code = "internal_error"
        retryable = False
    payload = _task_view_payload(view) if view is not None else {}
    payload["error"] = {
        "code": code,
        "retryable": bool(retryable),
        "message": SAFE_MESSAGES.get(code, "the proof-task command failed"),
    }
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), file=sys.stderr)


def _task_view_payload(view):
    """Every successful proof-task command returns exactly these fields."""
    return {
        "task_id": view["task_id"],
        "state": view["state"],
        "attempt": view["attempt"],
        "created_at": view["created_at"],
        "started_at": view["started_at"],
        "ended_at": view["ended_at"],
        "updated_at": view["updated_at"],
        "credential_sha256": view["credential_sha256"],
    }


def main():
    args = _parser().parse_args()
    try:
        if args.command == "infer":
            result = infer(json.loads(args.input.read_text(encoding="utf-8")))
        elif args.command == "demo":
            result = {name: infer(json.loads((ROOT / "samples" / f"{name}.json").read_text()))
                      for name in ("normal", "inspect")}
        elif args.command == "zk-setup":
            result = run_setup(args.model, args.dir)
        elif args.command == "zk-prove":
            result = run_prove(args.input, args.model, args.setup_dir, args.credential)
        elif args.command == "model-registry":
            if args.registry_command == "register":
                result = register_model(args.registry, args.version,
                                        args.manifest, args.model)
            elif args.registry_command == "list":
                result = {"models": list_models(args.registry)}
            elif args.registry_command == "enable":
                result = enable_model(args.registry, args.version)
            else:
                result = revoke_model(args.registry, args.version)
        elif args.command == "proof-task":
            # Every proof-task failure — expected or unexpected — becomes the
            # safe JSON envelope on stderr; no raw exception text or path may
            # ever reach the user.
            try:
                if args.task_command == "create":
                    view, _created = create_task(
                        args.store, args.input, args.model,
                        args.setup_dir, args.credential, args.idempotency_key)
                elif args.task_command == "status":
                    view = status_task(args.store, args.task_id)
                elif args.task_command == "run":
                    view = run_task(args.store, args.task_id)
                else:
                    view = retry_task(args.store, args.task_id)
            except TaskAttemptFailed as error:
                # The attempt was recorded as failed; the command fails and
                # emits one stderr JSON with the post-transition task state.
                _print_error(error, view=error.view)
                return 2
            except TaskError as error:
                _print_error(error)
                return 2
            except Exception:  # noqa: BLE001 - safety net, message is generic
                _print_error(RuntimeError("internal error"))
                return 2
            result = _task_view_payload(view)
        else:
            result = run_verify(args.credential, args.manifest, args.model,
                               args.settings, args.vk, args.srs,
                               args.registry, args.model_version)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except TaskError as error:
        _print_error(error)
        return 2
    except (ZkError, RegistryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
