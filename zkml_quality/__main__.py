import argparse
import json
import sys
from pathlib import Path

from .inference import ROOT, infer


def main():
    parser = argparse.ArgumentParser(description="Equipment-quality inference example")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("infer", help="Evaluate one JSON feature vector")
    command.add_argument("--input", required=True, type=Path)
    sub.add_parser("demo", help="Evaluate the bundled synthetic samples")
    args = parser.parse_args()
    try:
        if args.command == "infer":
            result = infer(json.loads(args.input.read_text(encoding="utf-8")))
        else:
            result = {name: infer(json.loads((ROOT / "samples" / f"{name}.json").read_text()))
                      for name in ("normal", "inspect")}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
