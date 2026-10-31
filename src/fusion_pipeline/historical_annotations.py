"""Manual historical annotation layer for co-authorship network figures.

The network, centrality scores, and Louvain communities remain computational
outputs. Laboratory/project callouts are a separate interpretive layer loaded
from a transparent CSV configuration. The configuration is period-specific and
can be replaced by a user-supplied file for other periods.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import textwrap

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ANNOTATION_COLUMNS = {
    "network", "start_year", "end_year", "label", "anchor_nodes",
    "label_dx_points", "label_dy_points", "provenance"
}
AUTHOR_LABEL_COLUMNS = {"network", "start_year", "end_year", "node"}


def load_period_annotations(
    path: str | Path,
    start_year: int,
    end_year: int,
    network: str = "coauthor_lcc",
) -> pd.DataFrame:
    """Load manual callouts that match an exact analytical period.

    Exact matching is deliberate: manual historical identification is never
    silently transferred to a different temporal window. Users studying another
    range can supply a custom annotation CSV using the same schema.
    """
    df = pd.read_csv(path)
    missing = ANNOTATION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing columns: {sorted(missing)}")
    out = df[
        df["network"].astype(str).eq(network)
        & df["start_year"].astype(int).eq(int(start_year))
        & df["end_year"].astype(int).eq(int(end_year))
    ].copy()
    return out.reset_index(drop=True)


def load_period_author_labels(
    path: str | Path | None,
    start_year: int,
    end_year: int,
    network: str = "coauthor_lcc",
) -> list[str]:
    if path is None:
        return []
    df = pd.read_csv(path)
    missing = AUTHOR_LABEL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Author-label CSV missing columns: {sorted(missing)}")
    out = df[
        df["network"].astype(str).eq(network)
        & df["start_year"].astype(int).eq(int(start_year))
        & df["end_year"].astype(int).eq(int(end_year))
    ]
    return out["node"].astype(str).tolist()


def parse_anchor_nodes(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(";") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def validate_annotation_anchors(G: nx.Graph, annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    nodes = set(G.nodes())
    for _, row in annotations.iterrows():
        requested = parse_anchor_nodes(row["anchor_nodes"])
        matched = [n for n in requested if n in nodes]
        missing = [n for n in requested if n not in nodes]
        rows.append({
            "label": row["label"],
            "requested_anchors": ";".join(requested),
            "matched_anchors": ";".join(matched),
            "missing_anchors": ";".join(missing),
            "n_requested": len(requested),
            "n_matched": len(matched),
            "status": "ok" if matched else "no_anchor_in_graph",
        })
    return pd.DataFrame(rows)


def _read_mapping_table(path: str | Path, value_candidates: Sequence[str]) -> dict:
    df = pd.read_csv(path)
    node_col = "node" if "node" in df.columns else "author" if "author" in df.columns else None
    if node_col is None:
        raise ValueError(f"No node/author column found in {path}")
    val_col = next((c for c in value_candidates if c in df.columns), None)
    if val_col is None:
        raise ValueError(f"None of {list(value_candidates)} found in {path}")
    return dict(zip(df[node_col].astype(str), df[val_col]))


def load_community_mapping(path: str | Path) -> dict[str, int]:
    return _read_mapping_table(path, ["community", "cluster"])


def load_betweenness_mapping(path: str | Path) -> dict[str, float]:
    return _read_mapping_table(path, ["betweenness_exact", "betweenness", "normalized_betweenness"])


def canonical_historical_layout(G: nx.Graph, seed: int = 42) -> dict:
    """Deterministic layout used for the annotated historical manuscript figures."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    iterations = 100 if n < 500 else 40 if m < 10000 else 20
    try:
        pos = nx.spring_layout(G, seed=seed, weight="weight", iterations=iterations, method="force")
    except TypeError:  # NetworkX versions before the method keyword.
        pos = nx.spring_layout(G, seed=seed, weight="weight", iterations=iterations)
    return {k: np.array([v[0] * 1.4, v[1]]) for k, v in pos.items()}


def _node_sizes(G: nx.Graph, betweenness: dict[str, float]) -> np.ndarray:
    vals = np.array([max(float(betweenness.get(n, 0.0)), 0.0) for n in G.nodes()], dtype=float)
    if vals.size == 0 or np.isclose(vals.max(), vals.min()):
        return np.full(len(vals), 20.0)
    scaled = (vals - vals.min()) / (vals.max() - vals.min() + 1e-15)
    return 7.0 + 210.0 * np.sqrt(scaled)


def _matched_anchor_xy(pos: dict, anchor_value: str) -> tuple[np.ndarray | None, list[str]]:
    anchors = [a for a in parse_anchor_nodes(anchor_value) if a in pos]
    if not anchors:
        return None, []
    return np.mean([pos[a] for a in anchors], axis=0), anchors


def plot_annotated_historical_lcc(
    G: nx.Graph,
    communities: dict[str, int],
    betweenness: dict[str, float],
    annotations: pd.DataFrame,
    start_year: int,
    end_year: int,
    output: str | Path,
    author_labels: Sequence[str] | None = None,
    annotation_mode: str = "auto",
    seed: int = 42,
    numbered_threshold: int = 9,
    export_layout_csv: str | Path | None = None,
    layout_override: dict[str, np.ndarray] | None = None,
    dpi: int = 300,
) -> Path:
    """Draw a canonical LCC and overlay a transparent manual lab/project layer.

    annotation_mode: ``boxes``, ``numbered``, ``auto``, or ``none``.
    In ``auto`` mode, dense periods use numbered markers plus a legend.
    """
    mode = annotation_mode.lower()
    if mode not in {"boxes", "numbered", "auto", "none"}:
        raise ValueError("annotation_mode must be boxes, numbered, auto, or none")
    if mode == "auto":
        mode = "numbered" if len(annotations) >= int(numbered_threshold) else "boxes"

    if layout_override is None:
        pos = canonical_historical_layout(G, seed=seed)
    else:
        missing=[n for n in G if n not in layout_override]
        if missing:
            raise ValueError(f"layout_override missing {len(missing)} graph nodes")
        pos={n: np.asarray(layout_override[n], dtype=float) for n in G}
    if export_layout_csv is not None:
        pd.DataFrame([{"node": n, "x": pos[n][0], "y": pos[n][1]} for n in G]).to_csv(export_layout_csv, index=False)

    sizes = _node_sizes(G, betweenness)
    author_labels = list(author_labels or [])
    if not author_labels:
        author_labels = sorted(G.nodes(), key=lambda n: float(betweenness.get(n, 0.0)), reverse=True)[:16]
    author_labels = [n for n in author_labels if n in G]

    period = str(start_year) if int(start_year) == int(end_year) else f"{start_year}-{end_year}"

    if mode == "numbered":
        # Dense historical callouts are placed in a two-column key below a
        # full-width graph so that labels remain legible at manuscript width.
        fig = plt.figure(figsize=(8, 6.8))
        ax = fig.add_axes([0.04, 0.34, 0.92, 0.57])
    else:
        fig, ax = plt.subplots(figsize=(8, 5))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.07, width=0.32, edge_color="0.45")
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes,
        node_color=[communities.get(n, 0) for n in G.nodes()],
        cmap=plt.get_cmap("tab20"), alpha=0.92, linewidths=0,
    )
    nx.draw_networkx_labels(
        G, pos, labels={n: n for n in author_labels},
        font_size=6.2 if G.number_of_nodes() > 500 else 7.2, ax=ax,
    )

    used_rows = []
    if mode == "boxes":
        for _, row in annotations.iterrows():
            xy, anchors = _matched_anchor_xy(pos, row["anchor_nodes"])
            if xy is None:
                continue
            used_rows.append((row, anchors))
            ax.annotate(
                str(row["label"]), xy=xy, xycoords="data",
                xytext=(float(row["label_dx_points"]), float(row["label_dy_points"])),
                textcoords="offset points", ha="center", va="center", fontsize=9.2,
                bbox=dict(boxstyle="round,pad=.26", fc="white", ec="0.25", lw=.75, alpha=.96),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="0.22", shrinkA=2, shrinkB=3, connectionstyle="arc3,rad=0.04"),
            )
    elif mode == "numbered":
        items = []
        for _, row in annotations.iterrows():
            xy, anchors = _matched_anchor_xy(pos, row["anchor_nodes"])
            if xy is None:
                continue
            used_rows.append((row, anchors))
            items.append((row, xy))
        for i, (row, xy) in enumerate(items, 1):
            ax.scatter([xy[0]], [xy[1]], s=260, facecolors="white", edgecolors="black", linewidths=1.0, zorder=8)
            ax.text(xy[0], xy[1], str(i), ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=9)
        fig.text(0.05, 0.295, "Manual laboratory/project identifications", fontsize=10.0, fontweight="bold", ha="left", va="top")
        n_items = len(items)
        split = (n_items + 1) // 2
        columns = [items[:split], items[split:]]
        starts = [0.05, 0.52]
        for col_idx, col_items in enumerate(columns):
            y = 0.255
            base_number = 1 if col_idx == 0 else split + 1
            for offset, (row, _) in enumerate(col_items):
                i = base_number + offset
                lab = str(row["label"])
                wrapped = textwrap.wrap(lab, width=37) or [lab]
                text = f"{i}. {wrapped[0]}"
                if len(wrapped) > 1:
                    text += "\n   " + "\n   ".join(wrapped[1:])
                fig.text(starts[col_idx], y, text, fontsize=9.0, ha="left", va="top", linespacing=1.0)
                y -= 0.053 * max(1, len(wrapped))

    ax.set_axis_off()
    ax.set_title(f"Largest connected component of the co-authorship network, {period}", fontsize=12.5, pad=8)
    if mode == "numbered":
        # The manuscript caption carries the encoding note; keeping the dense
        # key itself uncluttered improves legibility at journal page width.
        pass
    elif mode == "boxes":
        foot = "Node size: normalized betweenness centrality  •  Color: Louvain community  •  Boxes: manually identified laboratory/project contexts"
        ax.text(0.01, 0.01, foot, transform=ax.transAxes, fontsize=6.8, ha="left")
        fig.tight_layout()
    else:
        fig.tight_layout()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=.08)
    plt.close(fig)
    return out
