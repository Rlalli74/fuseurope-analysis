#!/usr/bin/env python3
"""Overlay manual laboratory/project identifications on a canonical LCC figure.

The computational graph is supplied as GraphML; communities and centrality are
supplied as canonical CSV outputs. The annotation CSV is an explicit historical
interpretive layer. It is matched to an exact user-selected start/end year.
"""
from pathlib import Path
import argparse
import sys
import networkx as nx
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from fusion_pipeline.historical_annotations import (
    load_period_annotations,
    load_period_author_labels,
    load_community_mapping,
    load_betweenness_mapping,
    validate_annotation_anchors,
    plot_annotated_historical_lcc,
)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("graphml", help="Canonical LCC GraphML for the selected period")
    ap.add_argument("--communities", required=True, help="CSV with node/community mapping")
    ap.add_argument("--centralities", required=True, help="CSV with betweenness or betweenness_exact")
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--annotation-csv", default=str(ROOT / "config" / "historical_lab_annotations.csv"))
    ap.add_argument("--author-label-csv", default=str(ROOT / "config" / "historical_figure_author_labels.csv"))
    ap.add_argument("--annotation-mode", choices=["auto", "boxes", "numbered", "none"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=300, help="Raster output resolution; ignored by vector formats")
    ap.add_argument("--output", required=True)
    ap.add_argument("--audit-csv", default=None)
    ap.add_argument("--layout-csv", default=None, help="Optional output path for the layout actually used")
    ap.add_argument("--layout-input-csv", default=None, help="Optional frozen node,x,y layout to reuse for manuscript continuity")
    ap.add_argument("--allow-no-configured-annotations", action="store_true")
    args = ap.parse_args()

    G = nx.read_graphml(args.graphml)
    ann = load_period_annotations(args.annotation_csv, args.start_year, args.end_year)
    if ann.empty and not args.allow_no_configured_annotations and args.annotation_mode != "none":
        raise SystemExit(
            "No manual annotation rows match this exact temporal range. "
            "Supply a custom --annotation-csv, use --annotation-mode none, or "
            "pass --allow-no-configured-annotations."
        )
    labels = load_period_author_labels(args.author_label_csv, args.start_year, args.end_year)
    communities = load_community_mapping(args.communities)
    betweenness = load_betweenness_mapping(args.centralities)
    layout_override = None
    if args.layout_input_csv:
        ld = pd.read_csv(args.layout_input_csv)
        required={"node","x","y"}
        if not required.issubset(ld.columns):
            raise SystemExit("--layout-input-csv must contain node,x,y columns")
        layout_override={str(r.node):(float(r.x),float(r.y)) for r in ld.itertuples(index=False)}

    audit = validate_annotation_anchors(G, ann)
    if args.audit_csv:
        Path(args.audit_csv).parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_csv, index=False)
    bad = audit[audit.status.ne("ok")] if not audit.empty else audit
    if not bad.empty:
        print("Warning: configured labels with no anchor in the supplied graph:")
        print(bad[["label", "requested_anchors"]].to_string(index=False))

    out = plot_annotated_historical_lcc(
        G, communities, betweenness, ann,
        args.start_year, args.end_year, args.output,
        author_labels=labels, annotation_mode=args.annotation_mode,
        seed=args.seed, export_layout_csv=args.layout_csv, layout_override=layout_override, dpi=args.dpi,
    )
    print(out)


if __name__ == "__main__":
    main()
