#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


def parse_judgement(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"yes", "no"}:
        return lowered
    matches = re.findall(r"judg(?:e)?ment\s*:\s*(yes|no)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return None


def iter_jsonl(path):
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in {path} at line {line_no}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-problem pass rates from judge output JSONL files."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory with output-*.jsonl (defaults to current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: input_dir/passrate_by_problem.json).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    jsonl_files = sorted(input_dir.glob("output-*.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No JSONL files found in {input_dir}")

    output_path = Path(args.output) if args.output else input_dir / "passrate_by_problem.json"

    stats = {}
    skipped = 0

    for jsonl_path in jsonl_files:
        for obj in iter_jsonl(jsonl_path):
            problem = obj.get("problem")
            expected = obj.get("expected_answer")
            judgement = parse_judgement(obj.get("judgement"))
            if problem is None or expected is None or judgement is None:
                skipped += 1
                continue

            predicted = obj.get("predicted_answer")
            key = (problem, expected)
            if key not in stats:
                stats[key] = {"yes": 0, "total": 0, "predicted": {}}
            stats[key]["total"] += 1
            if judgement == "yes":
                stats[key]["yes"] += 1

            if predicted is not None:
                pred_stats = stats[key]["predicted"].setdefault(
                    predicted, {"yes": 0, "total": 0}
                )
                pred_stats["total"] += 1
                if judgement == "yes":
                    pred_stats["yes"] += 1

    output = []
    for (problem, expected), counts in stats.items():
        total = counts["total"]
        pass_rate = counts["yes"] / total if total else 0.0
        predicted_list = []
        for predicted, pcounts in counts["predicted"].items():
            ptotal = pcounts["total"]
            predicted_list.append(
                {
                    "predicted_answer": predicted,
                    "correctness": pcounts["yes"] / ptotal if ptotal else 0.0,
                    "count": ptotal,
                }
            )
        predicted_list.sort(key=lambda item: item["correctness"], reverse=True)
        output.append(
            {
                "problem": problem,
                "expected_answer": expected,
                "pass_rate": pass_rate,
                "predicted_answers": predicted_list,
            }
        )

    with output_path.open("w") as f:
        json.dump(output, f, ensure_ascii=True, indent=2)
        f.write("\n")

    print(f"Wrote {len(output)} rows to {output_path}")
    if skipped:
        print(f"Skipped {skipped} records with missing/invalid fields", file=sys.stderr)


if __name__ == "__main__":
    main()
