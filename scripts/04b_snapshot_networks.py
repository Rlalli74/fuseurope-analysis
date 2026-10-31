#!/usr/bin/env python3
"""Detailed user-defined snapshot/period networks for historical reading."""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion_pipeline.exploration import (
    BASIC_GRAPH_METRICS, EXTENDED_GRAPH_METRICS, community_table,
    compute_bridge_scores, compute_centralities, compute_graph_metrics,
    parse_period_specs,
)
from fusion_pipeline.graph_export import (
    GraphExportOptions, enrich_projection_graph, write_export_audit,
    write_graphml_bundle,
)
from fusion_pipeline.networks import (
    apply_exact_period_record_exclusions, build_coauthorship_graph,
    filter_publications, largest_component_graph, prepare_network_dataframe,
)
from fusion_pipeline.socio_epistemic import load_titleword_config


DEFAULT_PERIODS = [
    "1959:1968", "1969:1973", "1974:1978", "1979", "1983", "1986",
    "1996", "2001",
]


def csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("phase3_csv")
    parser.add_argument("--output-dir", default="outputs_phase4b")
    parser.add_argument("--periods", nargs="*", default=DEFAULT_PERIODS, help="YEAR, START:END, or LABEL=START:END")
    parser.add_argument("--analyses", default="metrics,centralities,communities", help="metrics,centralities,communities,bridge_scores,graph_export")
    parser.add_argument("--metrics", default="extended")
    parser.add_argument("--centralities", default="degree,weighted_degree,fractional_degree,betweenness")
    parser.add_argument("--scope", choices=["lcc", "full"], default="lcc")
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--betweenness-exact-max-nodes", type=int, default=1000)
    parser.add_argument("--betweenness-samples", type=int, default=250)
    parser.add_argument("--document-type-mode", default="unrestricted_analytical_corpus")
    parser.add_argument("--max-team-size", type=int, default=None)
    parser.add_argument("--modularity-method", choices=["louvain", "greedy"], default="louvain")
    parser.add_argument("--historical-exclusions-csv", default=None, help="Retired compatibility hook; final runs leave this unset")
    parser.add_argument("--graph-attribute-profile", choices=["minimal", "manuscript", "exploration", "full"], default="exploration")
    parser.add_argument("--graph-formats", default="graphml,graphmlz")
    parser.add_argument("--graph-top-terms", type=int, default=8)
    parser.add_argument("--graph-max-node-titles", type=int, default=8)
    parser.add_argument("--graph-max-edge-titles", type=int, default=None)
    return parser


def export_period_graph(graph, publications, label, scope, start, end, args, out):
    exclusions, merge_map = load_titleword_config(ROOT / "config")
    options = GraphExportOptions(
        profile=args.graph_attribute_profile,
        top_terms=args.graph_top_terms,
        max_node_titles=args.graph_max_node_titles,
        max_edge_titles=args.graph_max_edge_titles,
        seed=args.seed,
        community_resolution=args.community_resolution,
        betweenness_exact_max_nodes=args.betweenness_exact_max_nodes,
        betweenness_samples=args.betweenness_samples,
    )
    rich, audit = enrich_projection_graph(
        graph, publications, entities_col="authors_network", options=options,
        title_exclusions=exclusions, title_merge_map=merge_map,
        graph_kind="author", start_year=start, end_year=end,
    )
    stem = out / f"{label}_{scope}_{args.graph_attribute_profile}"
    files = write_graphml_bundle(rich, stem, csv_list(args.graph_formats))
    audit.update(scope=scope, files=[str(path) for path in files])
    write_export_audit(audit, stem.with_name(stem.name + "_audit.json"))


def main():
    args = build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyses = set(csv_list(args.analyses))
    metrics = BASIC_GRAPH_METRICS if args.metrics == "basic" else EXTENDED_GRAPH_METRICS if args.metrics == "extended" else csv_list(args.metrics)
    centralities = csv_list(args.centralities)
    data = prepare_network_dataframe(pd.read_csv(args.phase3_csv, low_memory=False))
    metric_rows = []

    for label, start, end in parse_period_specs(args.periods):
        publications = filter_publications(
            data, start, end, args.max_team_size, args.document_type_mode,
        )
        excluded = pd.DataFrame()
        if args.historical_exclusions_csv:
            publications, excluded = apply_exact_period_record_exclusions(
                publications, args.historical_exclusions_csv, start, end,
            )
        graph = build_coauthorship_graph(publications)
        lcc = largest_component_graph(graph)
        if not excluded.empty:
            excluded.to_csv(out / f"{label}_record_exclusions_applied.csv", index=False)

        if "metrics" in analyses:
            row = compute_graph_metrics(
                graph, metrics, modularity_method=args.modularity_method,
                seed=args.seed,
            )
            row.update(period=label, start_year=start, end_year=end, papers=len(publications))
            metric_rows.append(row)
        if "centralities" in analyses:
            compute_centralities(
                graph, centralities, args.scope, seed=args.seed,
                betweenness_exact_max_nodes=args.betweenness_exact_max_nodes,
                betweenness_samples=args.betweenness_samples,
            ).to_csv(out / f"{label}_{args.scope}_centralities.csv", index=False)

        communities = None
        if "communities" in analyses or "bridge_scores" in analyses:
            communities = community_table(
                graph, args.scope, args.seed, resolution=args.community_resolution,
            )
            communities.to_csv(out / f"{label}_{args.scope}_communities.csv", index=False)
        if "bridge_scores" in analyses:
            compute_bridge_scores(
                graph, dict(zip(communities.node, communities.community)),
                args.scope, seed=args.seed,
                betweenness_samples=args.betweenness_samples,
                betweenness_exact_max_nodes=args.betweenness_exact_max_nodes,
            ).to_csv(out / f"{label}_{args.scope}_bridge_scores.csv", index=False)
        if "graph_export" in analyses:
            export_period_graph(graph, publications, label, "full", start, end, args, out)
            export_period_graph(lcc, publications, label, "lcc", start, end, args, out)
        print(label, "nodes=", graph.number_of_nodes(), "lcc=", lcc.number_of_nodes())

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(out / "selected_period_network_metrics.csv", index=False)
    summary = {
        "periods": args.periods,
        "analyses": sorted(analyses),
        "metrics": list(metrics),
        "centralities": centralities,
        "scope": args.scope,
        "document_type_mode": args.document_type_mode,
        "max_team_size": args.max_team_size,
        "modularity_method": args.modularity_method,
        "community_resolution": args.community_resolution,
        "seed": args.seed,
        "historical_exclusions_csv": args.historical_exclusions_csv,
        "graph_attribute_profile": args.graph_attribute_profile,
        "graph_formats": csv_list(args.graph_formats),
        "graph_text_summary_status": "descriptive exploration, not topic modelling",
    }
    (out / "phase4b_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
