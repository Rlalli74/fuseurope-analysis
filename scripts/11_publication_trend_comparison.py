#!/usr/bin/env python3
"""Compare fusion publication output with user-supplied annual baseline series."""
from pathlib import Path
import argparse, json, sys
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from fusion_pipeline.publication_trends import (
    annual_publication_counts, compare_publication_series,
    detect_common_trend_periods, trend_detection_sensitivity,
    plot_publication_comparison, save_publication_figure,
)


def parse_named_csvs(specs):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Baseline must be NAME=CSV: {spec}")
        name, path = spec.split("=", 1)
        out[name] = pd.read_csv(path)
    return out


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fusion", help="Annual fusion count CSV")
    src.add_argument("--fusion-corpus", help="Record-level canonical corpus CSV; annual counts are derived from its year column")
    ap.add_argument("--fusion-year-column", default="year")
    ap.add_argument("--baselines", nargs="+", required=True, help="NAME=CSV paths, e.g. physics=physics.csv engineering=engineering.csv")
    ap.add_argument("--analyses", nargs="+", choices=["comparison", "periods", "sensitivity", "figure"], default=["comparison", "periods", "sensitivity", "figure"])
    ap.add_argument("--output-dir", default="outputs_publication_trends")
    ap.add_argument("--start-year", type=int)
    ap.add_argument("--end-year", type=int)
    ap.add_argument("--plot-start-year", type=int)
    ap.add_argument("--plot-end-year", type=int)
    ap.add_argument("--smooth-window", type=int, default=2)
    ap.add_argument("--min-consecutive-years", type=int, default=3)
    ap.add_argument("--min-avg-slope", type=float, default=0.04)
    ap.add_argument("--exclude-years", nargs="*", type=int, default=[])
    ap.add_argument("--sensitivity-windows", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--sensitivity-slopes", nargs="*", type=float, default=[0.0, 0.02, 0.04, 0.06])
    ap.add_argument("--figure-formats", nargs="*", default=["png", "pdf", "tiff"])
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--no-shading", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.fusion:
        fusion = pd.read_csv(args.fusion)
        fusion_source = args.fusion
        fusion_mode = "annual_count_csv"
    else:
        records = pd.read_csv(args.fusion_corpus, low_memory=False)
        fusion = annual_publication_counts(records, args.fusion_year_column)
        fusion_source = args.fusion_corpus
        fusion_mode = "record_level_corpus"
        fusion.rename(columns={"count": "publications"}).to_csv(out / "fusion_annual_counts_derived.csv", index=False)

    series = {"fusion": fusion}
    series.update(parse_named_csvs(args.baselines))
    comp = compare_publication_series(series, "fusion", args.start_year, args.end_year, args.smooth_window)
    ratios = [f"fusion_vs_{name}" for name in series if name != "fusion"]
    periods = detect_common_trend_periods(
        comp, ratios, args.min_consecutive_years, args.min_avg_slope, args.exclude_years
    )

    if "comparison" in args.analyses:
        comp.to_csv(out / "publication_series_comparison.csv", index=False)
    if "periods" in args.analyses:
        periods.to_csv(out / "common_relative_trend_periods.csv", index=False)
    if "sensitivity" in args.analyses:
        sens = trend_detection_sensitivity(
            series, "fusion", args.start_year, args.end_year,
            smooth_windows=args.sensitivity_windows,
            min_avg_slopes=args.sensitivity_slopes,
            min_consecutive_years=args.min_consecutive_years,
            exclude_years=args.exclude_years,
        )
        sens.to_csv(out / "trend_detection_sensitivity.csv", index=False)
    figure_files = []
    if "figure" in args.analyses:
        fig, _ = plot_publication_comparison(
            comp, periods, focal_name="fusion",
            baseline_names=[n for n in series if n != "fusion"],
            plot_start_year=args.plot_start_year,
            plot_end_year=args.plot_end_year,
            title="Fusion publication output relative to Physics and Engineering",
            shade_periods=not args.no_shading,
        )
        figure_files = [str(p) for p in save_publication_figure(
            fig, out / "fusion_relative_publication_trends",
            formats=args.figure_formats, dpi=args.dpi
        )]

    summary = {
        "series": list(series),
        "fusion_source": fusion_source,
        "fusion_source_mode": fusion_mode,
        "analyses": args.analyses,
        "analysis_range": [args.start_year, args.end_year],
        "plot_range": [args.plot_start_year, args.plot_end_year],
        "smooth_window": args.smooth_window,
        "min_consecutive_years": args.min_consecutive_years,
        "min_avg_slope": args.min_avg_slope,
        "exclude_years": args.exclude_years,
        "figure_files": figure_files,
    }
    (out / "publication_trend_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
