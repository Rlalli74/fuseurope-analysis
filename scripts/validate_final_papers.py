#!/usr/bin/env python3
"""Compare private-run aggregate results with the frozen final-paper targets.

No record-level data are read by this validator.  A lawful full run must write
``final_acceptance_results.yaml`` (or JSON) below its external output/data root,
using the same nested keys as ``acceptance/FINAL_REFERENCE_VALUES.yaml``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_structured(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)


def _tolerance(path: tuple[str, ...], expected, meta: dict) -> float:
    if isinstance(expected, int) and not isinstance(expected, bool):
        return float(meta.get("integer_tolerance", 0))
    joined = ".".join(path).lower()
    if "pct" in joined or "percentage" in joined:
        return float(meta.get("displayed_percentage_tolerance_pp", 0.05))
    return float(meta.get("displayed_three_decimal_tolerance", 0.0005))


def compare(expected, actual, meta: dict, path=()):
    rows = []
    label = ".".join(map(str, path)) or "root"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{"path": label, "status": "FAIL", "expected": expected, "actual": actual, "reason": "type mismatch"}]
        for key, value in expected.items():
            if key not in actual:
                rows.append({"path": ".".join(map(str, path + (str(key),))), "status": "FAIL", "expected": value, "actual": None, "reason": "missing"})
            else:
                rows.extend(compare(value, actual[key], meta, path + (str(key),)))
        return rows
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [{"path": label, "status": "FAIL", "expected": expected, "actual": actual, "reason": "list length/type mismatch"}]
        for index, (left, right) in enumerate(zip(expected, actual)):
            rows.extend(compare(left, right, meta, path + (str(index),)))
        return rows
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            observed = float(actual)
            target = float(expected)
            tolerance = _tolerance(tuple(map(str, path)), expected, meta)
            delta = abs(observed - target)
            passed = math.isfinite(observed) and delta <= tolerance + 1e-12
            return [{"path": label, "status": "PASS" if passed else "FAIL", "expected": expected, "actual": actual, "tolerance": tolerance, "delta": delta}]
        except (TypeError, ValueError):
            return [{"path": label, "status": "FAIL", "expected": expected, "actual": actual, "reason": "not numeric"}]
    passed = actual == expected
    return [{"path": label, "status": "PASS" if passed else "FAIL", "expected": expected, "actual": actual, "tolerance": "exact"}]


def write_reports(rows, output_dir: Path, source: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(row["status"] == "PASS" for row in rows)
    failed = len(rows) - passed
    payload = {"status": "PASS" if failed == 0 else "FAIL", "checks": len(rows), "passed": passed, "failed": failed, "aggregate_results_source": str(source), "results": rows}
    (output_dir / "FINAL_ACCEPTANCE_REPORT.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Final numerical acceptance report", "", f"Status: **{payload['status']}**", "", f"Checks: {len(rows)}; passed: {passed}; failed: {failed}.", "", "Only aggregate generated results are compared; the validator never reads or packages record-level WoS data."]
    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        lines += ["", "## Failures", ""] + [f"- `{r['path']}`: expected `{r.get('expected')}`, observed `{r.get('actual')}` ({r.get('reason', 'outside tolerance')})" for r in failures]
    (output_dir / "FINAL_ACCEPTANCE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def resolve_results(args) -> Path:
    if args.results:
        return Path(args.results).expanduser().resolve()
    root = Path(args.data_root).expanduser().resolve()
    for name in ("final_acceptance_results.yaml", "final_acceptance_results.yml", "final_acceptance_results.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No aggregate acceptance result file found below {root}. Run the licensed_full_run workflow against lawful external WoS inputs and place final_acceptance_results.yaml there."
    )


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", help="External private run/output root")
    source.add_argument("--results", help="Generated aggregate YAML/JSON results")
    parser.add_argument("--reference", default=str(ROOT / "acceptance" / "FINAL_REFERENCE_VALUES.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "validation_reports"))
    args = parser.parse_args()
    try:
        result_path = resolve_results(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    reference = load_structured(Path(args.reference))
    generated = load_structured(result_path)
    expected = {key: value for key, value in reference.items() if key != "meta"}
    actual = {key: value for key, value in generated.items() if key != "meta"}
    rows = compare(expected, actual, reference.get("meta", {}))
    report = write_reports(rows, Path(args.output_dir), result_path)
    print(json.dumps({key: report[key] for key in ("status", "checks", "passed", "failed")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
