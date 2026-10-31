#!/usr/bin/env python3
"""Final ICR, co-country, and European-indicator workflow for the Lalli article."""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion_pipeline.exploration import parse_period_specs
from fusion_pipeline.internationalization import (
    cocountry_brokerage,
    cocountry_jet_geographic_benchmark,
    enlargement_definition_sensitivity,
    european_indicator_components,
    icr_with_pooled_smoothing,
    pool_european_indicators,
    prepare_country_dataframe,
    smooth_european_indicators,
)
from fusion_pipeline.networks import graph_to_graphml_safe


FINAL_ANALYSES = frozenset({"icr", "cocountry", "european"})


def csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("phase3_csv")
    parser.add_argument("--output-dir", default="outputs_phase5")
    parser.add_argument("--start-year", type=int, default=1973)
    parser.add_argument("--end-year", type=int, default=2007)
    parser.add_argument("--analyses", default="icr,cocountry,european", help="final choices: icr,cocountry,european")
    parser.add_argument("--network-periods", nargs="*", default=["1979:1982", "1986:1987", "1991:1995", "1996:2000"])
    parser.add_argument(
        "--use-geographic-country-network",
        action="store_true",
        help="Use geographic C1 union RP countries instead of the modified JET-node representation for co-country graphs",
    )
    args = parser.parse_args()

    analyses = set(csv_list(args.analyses))
    unknown = analyses - FINAL_ANALYSES
    if unknown:
        parser.error(f"unsupported analysis choices: {', '.join(sorted(unknown))}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_country_dataframe(pd.read_csv(args.phase3_csv, low_memory=False))

    if "icr" in analyses:
        annual, pooled = icr_with_pooled_smoothing(data, args.start_year, args.end_year, (3, 5))
        annual.to_csv(output / "yearly_ICR_geographic.csv", index=False)
        pooled[3].to_csv(output / "ICR_centered_3year_pooled.csv", index=False)
        pooled[5].to_csv(output / "ICR_centered_5year_pooled_sensitivity.csv", index=False)

    use_modified_jet = not args.use_geographic_country_network
    if "cocountry" in analyses:
        for label, start, end in parse_period_specs(args.network_periods):
            graph, lcc, nodes = cocountry_brokerage(
                data, start, end, modified_jet_representation=use_modified_jet
            )
            nodes.to_csv(output / f"cocountry_{label}_lcc_nodes.csv", index=False)
            cocountry_jet_geographic_benchmark(data, start, end).to_csv(
                output / f"cocountry_{label}_JET_geographic_benchmark.csv", index=False
            )
            graph_to_graphml_safe(graph, str(output / f"cocountry_{label}_full.graphml"))
            graph_to_graphml_safe(lcc, str(output / f"cocountry_{label}_lcc.graphml"))

    if "european" in analyses:
        annual_eu = european_indicator_components(data, start=args.start_year, end=args.end_year)
        annual_eu.to_csv(output / "european_indicators_annual_components.csv", index=False)
        principal_periods = [
            ("1973-1977", 1973, 1977), ("1978-1985", 1978, 1985),
            ("1986-1994", 1986, 1994), ("1995-1997", 1995, 1997),
            ("1998-2003", 1998, 2003), ("2004-2007", 2004, 2007),
        ]
        pool_european_indicators(annual_eu, principal_periods).to_csv(
            output / "european_indicators_principal_periods.csv", index=False
        )
        smooth_european_indicators(annual_eu, 3).to_csv(
            output / "european_indicators_centered_3year_pooled.csv", index=False
        )
        smooth_european_indicators(annual_eu, 5).to_csv(
            output / "european_indicators_centered_5year_pooled_sensitivity.csv", index=False
        )
        enlargement_definition_sensitivity(data).to_csv(
            output / "european_enlargement_definition_sensitivity.csv", index=False
        )

    summary = {
        "analysis_range": [args.start_year, args.end_year],
        "analyses": sorted(analyses),
        "network_periods": args.network_periods,
        "icr_representation": "geographic C1 union RP countries",
        "cocountry_representation": "modified JET node" if use_modified_jet else "strictly geographic",
        "cocountry_betweenness": "unweighted LCC topology",
    }
    (output / "phase5_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
