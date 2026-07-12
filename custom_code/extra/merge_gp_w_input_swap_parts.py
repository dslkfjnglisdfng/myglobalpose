#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from evaluate_gp_w_input_swap import aggregate, write_comparison


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("parts", nargs="+", type=Path); args = p.parse_args()
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for part in args.parts:
        for seq_dir in sorted(x for x in part.iterdir() if x.is_dir()):
            target = args.output_dir / seq_dir.name
            if target.exists():
                raise FileExistsError(target)
            shutil.copytree(seq_dir, target)
            rows.extend(json.loads(path.read_text()) for path in sorted(target.glob("G*.json")))
    agg = aggregate(rows)
    (args.output_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    write_comparison(args.dataset, rows, agg, args.output_dir / "comparison.csv")
    (args.output_dir / "merge_metadata.json").write_text(json.dumps({"dataset": args.dataset, "parts": [str(x) for x in args.parts], "sequences": sorted({r["sequence"] for r in rows})}, indent=2))


if __name__ == "__main__":
    main()
