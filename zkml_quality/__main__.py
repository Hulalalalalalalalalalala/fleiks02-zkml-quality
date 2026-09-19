import argparse
import json
import sys
from pathlib import Path

from .inference import ROOT, infer
from .prooftask import (
    ProofTaskError,
    create_task,
    retry_task,
    run_task,
    status_task,
)
from .registry import (
    RegistryError,
    enable_model,
    list_models,
    register_model,
    revoke_model,
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
        help="Manage the prover's offline proof-task library (create/status/run/retry)")
    task_sub = proof_task.add_subparsers(dest="task_command", required=True)

    create = task_sub.add_parser(
        "create", help="Validate a prove request and record a queued task")
    create.add_argument("--store", required=True, type=Path, help="Proof-task store directory")
    create.add_argument("--input", required=True, type=Path, help="JSON input containing features")
    create.add_argument("--model", required=True, type=Path, help="ONNX model used at setup")
    create.add_argument("--setup-dir", required=True, type=Path, help="Directory from zk-setup")
    create.add_argument("--credential", required=True, type=Path, help="Credential JSON to write")
    create.add_argument(
        "--idempotency-key", default=None,
        help="Reuse the original task when the identical request is resubmitted; "
             "a different request with the same key is rejected")

    for name, help_text in (
            ("status", "Print the current task state without modifying the store"),
            ("run", "Execute one queued task with a real EZKL proof"),
            ("retry", "Re-queue a failed task whose error is retryable")):
        command = task_sub.add_parser(name, help=help_text)
        command.add_argument("--store", required=True, type=Path,
                             help="Proof-task store directory")
        command.add_argument("--task-id", required=True, help="Task id (pt-...)")
    return parser


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
            if args.task_command == "create":
                result = create_task(args.store, args.input, args.model,
                                     args.setup_dir, args.credential,
                                     args.idempotency_key)
            elif args.task_command == "status":
                result = status_task(args.store, args.task_id)
            elif args.task_command == "run":
                result = run_task(args.store, args.task_id)
            else:
                result = retry_task(args.store, args.task_id)
        else:
            result = run_verify(args.credential, args.manifest, args.model,
                               args.settings, args.vk, args.srs,
                               args.registry, args.model_version)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (ZkError, RegistryError, ProofTaskError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
