import argparse
import json
import sys
from pathlib import Path

from .inference import ROOT, infer
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
        help="Verify a credential against a verifier-owned manifest; no input, proving key or network")
    verify.add_argument("--credential", required=True, type=Path)
    verify.add_argument(
        "--manifest", required=True, type=Path,
        help="manifest.json from a zk-setup run the verifier independently obtained and trusts")
    verify.add_argument("--model", required=True, type=Path, help="Verifier-chosen ONNX model")
    verify.add_argument("--settings", required=True, type=Path)
    verify.add_argument("--vk", required=True, type=Path, help="Verification key")
    verify.add_argument("--srs", required=True, type=Path)
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
        else:
            result = run_verify(args.credential, args.manifest, args.model,
                               args.settings, args.vk, args.srs)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except ZkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
