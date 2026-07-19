# type: ignore
"""Generate a deterministic, strictly unseen full-width calculator test set."""

import argparse
import collections
import json
import random
from pathlib import Path

import yaml

from generate_data_v2 import (
    SCENARIO_FUNCTIONS,
    apportion,
    build_central_thresholds,
    build_rows,
    central_band,
    load_excluded_jsonl,
    load_reserved_prompts,
    load_spec,
    split_micro_categories,
    supports_commutative_swap,
)


def cell_totals(total):
    cells = collections.OrderedDict(
        ((left, right), 1.0)
        for left in range(1, 10)
        for right in range(1, 10))
    return collections.OrderedDict(apportion(total, cells))


def strata_for_operation(op, total, spec):
    cells = cell_totals(total)
    if op == "*":
        central = split_micro_categories(
            cells, collections.OrderedDict((band, 0.25) for band in range(4)))
        central_cells = collections.OrderedDict(
            ((left, right, band), count)
            for (left, right), bands in central.items()
            for band, count in bands.items()
            if count)
        signs = spec["operations"][op]["sign_pattern_weights"]
        joint = split_micro_categories(central_cells, signs)
        return collections.OrderedDict(
            ((left, right, band, sign), count)
            for (left, right, band), sign_counts in joint.items()
            for sign, count in sign_counts.items()
            if count)

    if op == "/":
        signs = spec["operations"][op]["sign_pattern_weights"]
    else:
        negative = float(spec["domain"]["negative_first_operand_ratio"])
        signs = collections.OrderedDict(
            (("positive", 1.0 - negative), ("negative", negative)))
    joint = split_micro_categories(cells, signs)
    return collections.OrderedDict(
        ((left, right, sign), count)
        for (left, right), sign_counts in joint.items()
        for sign, count in sign_counts.items()
        if count)


def signed_operands(op, left, right, sign):
    if op == "*":
        negative_a = sign in ("negative_positive", "negative_negative")
        negative_b = sign == "negative_negative"
    else:
        negative_a = sign == "negative"
        negative_b = False
    for magnitude_a in range(left * 100, left * 100 + 100):
        a = -magnitude_a if negative_a else magnitude_a
        for magnitude_b in range(right * 100, right * 100 + 100):
            b = -magnitude_b if negative_b else magnitude_b
            yield a, b


def generate_test(spec, rows_per_operation, seed, excluded):
    rng = random.Random(seed)
    thresholds = build_central_thresholds()
    selected = set()
    samples = []
    for op in spec["domain"]["operations"]:
        for stratum, count in strata_for_operation(
                op, rows_per_operation, spec).items():
            left, right = stratum[:2]
            band = stratum[2] if op == "*" else None
            sign = stratum[3] if op == "*" else stratum[2]
            candidates = []
            for a, b in signed_operands(op, left, right, sign):
                key = (a, b, op)
                swapped = (b, a, op)
                if key in excluded or key in selected:
                    continue
                if (supports_commutative_swap(a, b, op)
                        and swapped in selected):
                    continue
                if op == "*" and central_band(a, b, thresholds) != band:
                    continue
                candidates.append((a, b))
            if len(candidates) < count:
                raise RuntimeError(
                    f"unseen stratum {op} {stratum} has "
                    f"{len(candidates)} candidates for {count} rows")
            for a, b in rng.sample(candidates, count):
                selected.add((a, b, op))
                samples.append({
                    "a": a,
                    "b": b,
                    "op": op,
                    "scenario": SCENARIO_FUNCTIONS[op](a, b),
                    "primary": (
                        ("mul3", left, right, band, sign) if op == "*" else
                        ("div3", left, right, sign) if op == "/" else
                        ("add3", left, right) if op == "+" else
                        ("sub3", left, right, sign)),
                })
    rng.shuffle(samples)
    rows = build_rows(
        samples, "test", reverse=True,
        max_seq_len=int(spec["validation"]["require_supported_sequence_length"]),
        division_answer_format=spec["domain"].get(
            "division_answer_format", "fixed_width"),
        representation=spec["domain"].get("representation", "abacus-v1"),
        division_focus_scenarios=spec["operations"]["/"].get(
            "capability_focus_scenarios", []))
    return samples, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="config/data_distribution_v2.yaml")
    parser.add_argument("--exclude-jsonl", action="append", required=True)
    parser.add_argument("--rows-per-operation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--out", default="sample/unseen_test_v2_seed20260716.jsonl")
    parser.add_argument(
        "--report-out",
        default="sample/unseen_test_v2_seed20260716_report.json")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    excluded = load_reserved_prompts(spec)
    excluded.update(load_excluded_jsonl(args.exclude_jsonl))
    samples, rows = generate_test(
        spec, args.rows_per_operation, args.seed, excluded)
    keys = {(sample["a"], sample["b"], sample["op"]) for sample in samples}
    if keys & excluded:
        raise AssertionError("unseen test overlaps an excluded prompt")
    if len(keys) != len(samples):
        raise AssertionError("unseen test contains duplicate prompts")

    operation_counts = collections.Counter(s["op"] for s in samples)
    scenario_counts = collections.Counter(
        f"{s['op']}|{s['scenario']}" for s in samples)
    sign_counts = collections.Counter(
        ("*|" + row["multiplication_sign_pattern"] if row["operation"] == "*"
         else row["operation"] + "|" +
         ("negative" if row["first_operand_negative"] else "positive"))
        for row in rows)
    central_counts = collections.Counter(
        row["central_total_band"] for row in rows if row["operation"] == "*")
    cell_counts = collections.defaultdict(collections.Counter)
    for row in rows:
        cell_counts[row["operation"]][row["hundreds_cell"]] += 1
    report = {
        "name": (
            f"three_digit_full_width_unseen_test_v"
            f"{spec.get('version', 2)}"),
        "representation": spec["domain"].get(
            "representation", "abacus-v1"),
        "division_answer_format": spec["domain"].get(
            "division_answer_format", "fixed_width"),
        "seed": args.seed,
        "rows": len(rows),
        "rows_per_operation": dict(operation_counts),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "sign_counts": dict(sorted(sign_counts.items())),
        "multiplication_central_bands": dict(sorted(central_counts.items())),
        "hundreds_cell_min_max": {
            op: {"min": min(counts.values()), "max": max(counts.values())}
            for op, counts in cell_counts.items()},
        "integrity": {
            "unique_prompts": len(keys),
            "excluded_prompt_overlap": 0,
            "all_operands_three_digits": all(
                100 <= abs(s["a"]) <= 999 and 100 <= abs(s["b"]) <= 999
                for s in samples),
        },
    }
    out = Path(args.out)
    report_out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with report_out.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {len(rows):,} unseen test rows -> {out}")


if __name__ == "__main__":
    main()
