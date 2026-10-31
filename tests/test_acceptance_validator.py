import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_final_papers.py"
SPEC = importlib.util.spec_from_file_location("validate_final_papers", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_acceptance_comparison_applies_exact_integer_and_percentage_tolerances():
    meta = {"integer_tolerance": 0, "displayed_percentage_tolerance_pp": 0.05, "displayed_three_decimal_tolerance": 0.0005}
    expected = {"records": 10, "lcc_share_pct": 19.2, "centrality": 0.291, "terms": ["a", "b"]}
    actual = {"records": 10, "lcc_share_pct": 19.249, "centrality": 0.2914, "terms": ["a", "b"]}
    rows = MODULE.compare(expected, actual, meta)
    assert all(row["status"] == "PASS" for row in rows)


def test_acceptance_comparison_fails_missing_and_outside_tolerance_values():
    meta = {"integer_tolerance": 0, "displayed_percentage_tolerance_pp": 0.05, "displayed_three_decimal_tolerance": 0.0005}
    rows = MODULE.compare({"records": 10, "score": 0.291}, {"records": 9}, meta)
    assert {row["status"] for row in rows} == {"FAIL"}
    assert {row["path"] for row in rows} == {"records", "score"}
