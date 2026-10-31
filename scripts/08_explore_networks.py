#!/usr/bin/env python3
"""Parameter-driven author and co-country network explorer.

The explorer is intentionally broader than the analyses reported in either
article. Named GraphML profiles keep manuscript reproduction separate from
rich historical exploration while retaining publication-level provenance.
"""
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion_pipeline.exploration import (
    BASIC_GRAPH_METRICS, EXTENDED_GRAPH_METRICS, centrality_over_time,
    community_table, compute_bridge_scores, longitudinal_author_metrics,
    make_time_windows, parse_period_specs,
)
from fusion_pipeline.graph_export import (
    GraphExportOptions, enrich_projection_graph, write_export_audit,
    write_graphml_bundle,
)
from fusion_pipeline.internationalization import (
    build_cocountry_graph, cocountry_longitudinal_metrics,
    country_centrality_over_time, prepare_country_dataframe,
)
from fusion_pipeline.networks import (
    apply_exact_period_record_exclusions, build_coauthorship_graph,
    filter_publications, largest_component_graph, prepare_network_dataframe,
)
from fusion_pipeline.socio_epistemic import load_titleword_config
from fusion_pipeline.visualization import plot_longitudinal_metrics, plot_network


def csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def graph_options(args):
    return GraphExportOptions(
        profile=args.graph_attribute_profile,
        node_attributes=tuple(csv_list(args.graph_node_attributes)),
        edge_attributes=tuple(csv_list(args.graph_edge_attributes)),
        top_terms=args.graph_top_terms,
        topic_label_terms=args.graph_topic_label_terms,
        max_node_titles=args.graph_max_node_titles,
        max_edge_titles=args.graph_max_edge_titles,
        include_bigrams=not args.graph_unigrams_only,
        text_source=args.graph_text_source,
        seed=args.seed,
        community_resolution=args.community_resolution,
        betweenness_exact_max_nodes=args.betweenness_exact_max_nodes,
        betweenness_samples=args.betweenness_samples,
    )


def export_graph(graph, publications, entities_col, graph_kind, label, start, end, args, out):
    selected_graph = largest_component_graph(graph) if args.graph_scope == "lcc" else graph
    exclusions, merge_map = load_titleword_config(ROOT / "config")
    enriched, audit = enrich_projection_graph(
        selected_graph, publications, entities_col=entities_col,
        options=graph_options(args), title_exclusions=exclusions,
        title_merge_map=merge_map, graph_kind=graph_kind,
        start_year=start, end_year=end,
    )
    stem = out / f"{graph_kind}_{label}_{args.graph_scope}_{args.graph_attribute_profile}"
    written = write_graphml_bundle(enriched, stem, csv_list(args.graph_formats))
    audit.update({"files": [str(path) for path in written], "scope": args.graph_scope})
    write_export_audit(audit, stem.with_name(stem.name + "_audit.json"))


def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("phase3_csv")
    parser.add_argument("--output-dir", default="outputs_exploration")
    parser.add_argument("--network", choices=["author", "country"], default="author")
    parser.add_argument("--analyses", default="metrics", help="metrics,centralities,communities,bridge_scores,graph_export")
    parser.add_argument("--start-year", type=int, default=1956)
    parser.add_argument("--end-year", type=int, default=2007)
    parser.add_argument("--window-mode", choices=["annual", "rolling", "period"], default="annual")
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--metrics", default="basic", help="basic, extended, or comma-separated names")
    parser.add_argument("--centralities", default="degree,degree_normalized,weighted_degree,fractional_degree,betweenness,closeness,eigenvector")
    parser.add_argument("--scope", choices=["full", "lcc"], default="lcc", help="Scope for tables and calculations")
    parser.add_argument("--focal-nodes", default="")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--document-type-mode", default="all_selected")
    parser.add_argument("--max-team-size", type=int, default=None)
    parser.add_argument("--modularity-method", choices=["louvain", "greedy"], default="louvain")
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--betweenness-exact-max-nodes", type=int, default=1000)
    parser.add_argument("--betweenness-samples", type=int, default=250)
    parser.add_argument("--plot-metrics", default="")
    parser.add_argument("--plot-start-year", type=int, default=None)
    parser.add_argument("--plot-end-year", type=int, default=None)
    parser.add_argument("--plot-network-period", default="", help="YEAR or START:END")
    parser.add_argument("--node-size-by", default="betweenness")
    parser.add_argument("--label-top-n", type=int, default=20)
    parser.add_argument("--max-plot-nodes", type=int, default=None)
    parser.add_argument("--layout", choices=["spring", "kamada_kawai", "circular"], default="spring")
    parser.add_argument("--geographic-countries", action="store_true")

    group = parser.add_argument_group("rich GraphML/GraphMLZ export")
    group.add_argument("--graph-attribute-profile", choices=["minimal", "manuscript", "exploration", "full", "custom"], default="exploration")
    group.add_argument("--graph-node-attributes", default="", help="Comma-separated override")
    group.add_argument("--graph-edge-attributes", default="", help="Comma-separated override; titles describe only publications supporting that link")
    group.add_argument("--graph-formats", default="graphml,graphmlz")
    group.add_argument("--graph-scope", choices=["full", "lcc"], default="lcc")
    group.add_argument("--graph-top-terms", type=int, default=8)
    group.add_argument("--graph-topic-label-terms", type=int, default=3)
    group.add_argument("--graph-max-node-titles", type=int, default=8)
    group.add_argument("--graph-max-edge-titles", type=int, default=None, help="Default keeps every edge-supporting title")
    group.add_argument("--graph-unigrams-only", action="store_true")
    group.add_argument("--graph-text-source", choices=["title", "title_keywords"], default="title_keywords")
    group.add_argument("--apply-exact-period-exclusions", action="store_true", help="Apply close-reading exclusions only when a window matches the registered period exactly")
    group.add_argument("--historical-exclusions-csv", default=str(ROOT / "config" / "historical_record_exclusions.csv"))
    return parser


def author_analyses(df, args, analyses, metrics, centralities, focal, out):
    if "metrics" in analyses:
        longitudinal_author_metrics(
            df, args.start_year, args.end_year, args.window_mode,
            args.window_size, args.step, metrics, args.document_type_mode,
            args.max_team_size, args.modularity_method, args.seed,
        ).to_csv(out / "author_longitudinal_metrics.csv", index=False)
    if "centralities" in analyses:
        centrality_over_time(
            df, args.start_year, args.end_year, centralities, focal or None,
            args.top_n if not focal else None, args.window_mode,
            args.window_size, args.step, args.scope, args.document_type_mode,
            args.max_team_size, args.seed, args.betweenness_exact_max_nodes,
            args.betweenness_samples,
        ).to_csv(out / "author_centrality_over_time.csv", index=False)
    if not analyses & {"communities", "bridge_scores", "graph_export"}:
        return
    for window in make_time_windows(args.start_year, args.end_year, args.window_mode, args.window_size, args.step):
        sub = filter_publications(df, window.start_year, window.end_year, args.max_team_size, args.document_type_mode)
        if args.apply_exact_period_exclusions:
            sub, rules = apply_exact_period_record_exclusions(
                sub, args.historical_exclusions_csv, window.start_year, window.end_year,
            )
            if not rules.empty:
                rules.to_csv(out / f"author_{window.label}_record_exclusions_applied.csv", index=False)
        graph = build_coauthorship_graph(sub)
        communities = None
        if "communities" in analyses or "bridge_scores" in analyses:
            communities = community_table(graph, args.scope, args.seed, resolution=args.community_resolution)
            communities.to_csv(out / f"author_{window.label}_{args.scope}_communities.csv", index=False)
        if "bridge_scores" in analyses:
            compute_bridge_scores(
                graph, dict(zip(communities.node, communities.community)), args.scope,
                seed=args.seed, betweenness_samples=args.betweenness_samples,
                betweenness_exact_max_nodes=args.betweenness_exact_max_nodes,
            ).to_csv(out / f"author_{window.label}_{args.scope}_bridge_scores.csv", index=False)
        if "graph_export" in analyses:
            export_graph(graph, sub, "authors_network", "author", window.label, window.start_year, window.end_year, args, out)


def country_analyses(df, args, analyses, metrics, centralities, focal, out):
    use_network = not args.geographic_countries
    if "metrics" in analyses:
        cocountry_longitudinal_metrics(
            df, args.start_year, args.end_year, args.window_mode,
            args.window_size, args.step, metrics, use_network,
            args.modularity_method, args.seed,
        ).to_csv(out / "country_longitudinal_metrics.csv", index=False)
    if "centralities" in analyses:
        country_centrality_over_time(
            df, args.start_year, args.end_year, centralities, focal or None,
            args.top_n if not focal else None, args.window_mode,
            args.window_size, args.step, args.scope, use_network, args.seed,
            args.betweenness_exact_max_nodes, args.betweenness_samples,
        ).to_csv(out / "country_centrality_over_time.csv", index=False)
    if not analyses & {"communities", "bridge_scores", "graph_export"}:
        return
    entities_col = "countries_network" if use_network else "countries_normalized"
    for window in make_time_windows(args.start_year, args.end_year, args.window_mode, args.window_size, args.step):
        sub = df[df.year.between(window.start_year, window.end_year)].copy()
        graph = build_cocountry_graph(df, window.start_year, window.end_year, use_network)
        communities = None
        if "communities" in analyses or "bridge_scores" in analyses:
            communities = community_table(graph, args.scope, args.seed, resolution=args.community_resolution)
            communities.to_csv(out / f"country_{window.label}_{args.scope}_communities.csv", index=False)
        if "bridge_scores" in analyses:
            compute_bridge_scores(
                graph, dict(zip(communities.node, communities.community)), args.scope,
                seed=args.seed, betweenness_samples=args.betweenness_samples,
                betweenness_exact_max_nodes=args.betweenness_exact_max_nodes,
            ).to_csv(out / f"country_{window.label}_{args.scope}_bridge_scores.csv", index=False)
        if "graph_export" in analyses:
            export_graph(graph, sub, entities_col, "country", window.label, window.start_year, window.end_year, args, out)


def main():
    args = build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analyses = set(csv_list(args.analyses))
    centralities = csv_list(args.centralities)
    focal = csv_list(args.focal_nodes)
    metrics = BASIC_GRAPH_METRICS if args.metrics == "basic" else EXTENDED_GRAPH_METRICS if args.metrics == "extended" else csv_list(args.metrics)
    raw = pd.read_csv(args.phase3_csv, low_memory=False)
    if args.network == "author":
        df = prepare_network_dataframe(raw)
        author_analyses(df, args, analyses, metrics, centralities, focal, out)
    else:
        df = prepare_country_dataframe(raw)
        country_analyses(df, args, analyses, metrics, centralities, focal, out)

    plot_metrics = csv_list(args.plot_metrics)
    metrics_path = out / f"{args.network}_longitudinal_metrics.csv"
    if plot_metrics and metrics_path.exists():
        plot_longitudinal_metrics(pd.read_csv(metrics_path), plot_metrics, args.plot_start_year, args.plot_end_year, out / "figures")
    if args.plot_network_period:
        _, start, end = parse_period_specs([args.plot_network_period])[0]
        if args.network == "author":
            graph = build_coauthorship_graph(filter_publications(df, start, end, args.max_team_size, args.document_type_mode))
        else:
            graph = build_cocountry_graph(df, start, end, not args.geographic_countries)
        plot_network(
            graph, args.scope, args.node_size_by, "community", args.label_top_n,
            0, args.max_plot_nodes, args.layout, args.seed,
            args.community_resolution, args.betweenness_exact_max_nodes,
            args.betweenness_samples, out / "figures" / f"{args.network}_{start}-{end}_{args.node_size_by}.png",
        )

    summary = {
        "network": args.network,
        "analyses": sorted(analyses),
        "analysis_range": [args.start_year, args.end_year],
        "window_mode": args.window_mode,
        "window_size": args.window_size,
        "step": args.step,
        "metrics": list(metrics),
        "centralities": centralities,
        "scope": args.scope,
        "graph_scope": args.graph_scope,
        "graph_attribute_profile": args.graph_attribute_profile,
        "graph_formats": csv_list(args.graph_formats),
        "graph_text_source": args.graph_text_source,
        "graph_text_summary_status": "descriptive exploration, not topic modelling",
        "document_type_mode": args.document_type_mode,
        "max_team_size": args.max_team_size,
        "exact_period_exclusions_enabled": args.apply_exact_period_exclusions,
    }
    (out / "exploration_run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
