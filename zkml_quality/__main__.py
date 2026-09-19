import argparse
import json
import sys
from pathlib import Path

from .inference import ROOT, infer
from .zk import ZkError, run_prove, run_setup, run_verify


def _add_model(parser):
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to the ONNX quality model")


def main():
    parser = argparse.ArgumentParser(description="Equipment-quality inference example")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("infer", help="Evaluate one JSON feature vector")
    command.add_argument("--input", required=True, type=Path)

    sub.add_parser("demo", help="Evaluate the bundled synthetic samples")

    setup_parser = sub.add_parser("zk-setup",
                                  help="Compile the circuit and generate SRS, proving and verifying keys")
    _add_model(setup_parser)
    setup_parser.add_argument("--dir", required=True, type=Path,
                              help="Empty directory that receives the setup artifacts")

    prove_parser = sub.add_parser("zk-prove",
                                  help="Prove private features and write a JSON credential")
    prove_parser.add_argument("--input", required=True, type=Path,
                              help="JSON input containing six features in [0, 1]")
    _add_model(prove_parser)
    prove_parser.add_argument("--setup-dir", required=True, type=Path)
    prove_parser.add_argument("--credential", required=True, type=Path,
                              help="Destination path for the proof credential JSON")

    verify_parser = sub.add_parser("zk-verify",
                                   help="Verify a credential with verifier-chosen artifacts (no input or pk)")
    verify_parser.add_argument("--credential", required=True, type=Path)
    _add_model(verify_parser)
    verify_parser.add_argument("--settings", required=True, type=Path)
    verify_parser.add_argument("--vk", required=True, type=Path)
    verify_parser.add_argument("--srs", required=True, type=Path)

    args = parser.parse_args()
    try:
        if args.command == "infer":
            result = infer(json.loads(args.input.read_text(encoding="utf-8")))
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        elif args.command == "demo":
            result = {name: infer(json.loads((ROOT / "samples" / f"{name}.json").read_text()))
                      for name in ("normal", "inspect")}
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        elif args.command == "zk-setup":
            result = run_setup(args.model, args.dir)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
        elif args.command == "zk-prove":
            result = run_prove(args.input, args.model, args.setup_dir, args.credential)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True))
        else:
            result = run_verify(args.credential, args.model, args.settings, args.vk, args.srs)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    except (ValueError, OSError, RuntimeError, ZkError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
