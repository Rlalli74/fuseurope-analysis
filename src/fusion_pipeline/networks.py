"""Canonical network construction and robustness utilities for the FusEurope corpus.

This module builds paper-projection co-authorship networks from the phase-3
``authors_network`` field. It keeps ordinary paper-count edge weights and an
additional Newman-style fractional collaboration weight (1/(n_authors-1) per
coauthor pair contribution). Connectivity analyses use the unweighted graph;
fractional weights are available for weighted-degree robustness checks.

The module deliberately exposes filters for team size and WoS document type so
that the sensitivity analyses requested by referees can be reproduced without
modifying the canonical corpus.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence
import ast
import json
import math

import numpy as np
import pandas as pd
import networkx as nx


def parse_list_cell(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [text]


def prepare_network_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["authors_network", "authors_disambiguated", "authors_normalized", "document_types"]:
        if col in out.columns:
            out[col] = out[col].apply(parse_list_cell)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out[out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    if "authors_network" in out.columns:
        out["team_size"] = out["authors_network"].apply(len)
    return out


def apply_exact_period_record_exclusions(
    df: pd.DataFrame,
    exclusions_csv: str | None,
    start_year: int,
    end_year: int,
    analysis: str = "coauthor_snapshot",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply provenance-tracked exclusions scoped to one exact analysis period.

    The source corpus is never altered.  A row is active only when analysis and
    both period endpoints match, preventing a historical close-reading decision
    from leaking into annual trends or other network windows.
    """
    if not exclusions_csv:
        return df.copy(), pd.DataFrame()
    rules = pd.read_csv(exclusions_csv)
    active = rules[
        rules.analysis.eq(analysis)
        & rules.start_year.eq(int(start_year))
        & rules.end_year.eq(int(end_year))
        & rules.scope.eq("exact_period_only")
    ].copy()
    if active.empty:
        return df.copy(), active
    id_col = "record_id" if "record_id" in df.columns else "ut"
    out = df[~df[id_col].astype(str).isin(active.record_id.astype(str))].copy()
    return out, active


def _doc_type_allowed(types: Sequence[str], mode: str) -> bool:
    types = set(types or [])
    if mode in {"all_selected", "unrestricted_analytical_corpus"}:
        return True
    if mode == "exclude_meeting_abstracts":
        return "Meeting Abstract" not in types
    if mode == "exclude_meeting_abstracts_editorials":
        # Backward-compatible input label. Editorial Material must already be
        # absent from the final analytical corpus.
        return not bool(types & {"Meeting Abstract", "Editorial Material"})
    if mode in {"exclude_proceedings_meeting_abstracts", "exclude_proceedings_and_meeting_abstracts"}:
        return not bool(types & {"Proceedings Paper", "Meeting Abstract"})
    if mode == "articles_only":
        return types == {"Article"}
    raise ValueError(f"Unknown document-type mode: {mode}")


def filter_publications(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
    max_team_size: int | None = None,
    document_type_mode: str = "all_selected",
) -> pd.DataFrame:
    out = df
    if start_year is not None:
        out = out[out["year"] >= int(start_year)]
    if end_year is not None:
        out = out[out["year"] <= int(end_year)]
    if max_team_size is not None:
        out = out[out["team_size"] <= int(max_team_size)]
    if document_type_mode != "all_selected":
        out = out[out["document_types"].apply(lambda x: _doc_type_allowed(x, document_type_mode))]
    return out.copy()


def build_coauthorship_graph(
    df: pd.DataFrame,
    authors_col: str = "authors_network",
    store_record_ids: bool = False,
) -> nx.Graph:
    """Project papers to an undirected author graph.

    ``paper_count`` counts joint papers. ``weight`` aliases paper_count for
    compatibility with NetworkX. ``fractional_weight`` adds 1/(n-1) for each
    pair on an n-author paper (Newman-style collaboration weighting).
    """
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = row.get(authors_col, [])
        if not isinstance(authors, list):
            authors = parse_list_cell(authors)
        authors = list(dict.fromkeys(a for a in authors if a))
        n = len(authors)
        if n == 0:
            continue
        rid = str(row.get("record_id", ""))
        for a in authors:
            if a not in G:
                G.add_node(a, label=a, paper_count=0)
            G.nodes[a]["paper_count"] += 1
        if n < 2:
            continue
        frac = 1.0 / (n - 1)
        for a, b in combinations(authors, 2):
            if not G.has_edge(a, b):
                attrs = {"paper_count": 0, "weight": 0, "fractional_weight": 0.0}
                if store_record_ids:
                    attrs["record_ids"] = []
                G.add_edge(a, b, **attrs)
            G[a][b]["paper_count"] += 1
            G[a][b]["weight"] += 1
            G[a][b]["fractional_weight"] += frac
            if store_record_ids:
                G[a][b]["record_ids"].append(rid)
    return G


def largest_component_graph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return nx.Graph()
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def graph_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {
            "nodes": 0, "edges": 0, "mean_degree": 0.0, "density": 0.0,
            "transitivity": np.nan, "average_clustering": np.nan,
            "components": 0, "lcc_nodes": 0, "lcc_edges": 0,
            "lcc_share_nodes": 0.0, "lcc_share_edges": 0.0,
            "lcc_density": np.nan, "lcc_transitivity": np.nan,
            "lcc_average_clustering": np.nan,
        }
    L = largest_component_graph(G)
    ln, lm = L.number_of_nodes(), L.number_of_edges()
    return {
        "nodes": n,
        "edges": m,
        "mean_degree": (2.0 * m / n) if n else 0.0,
        "density": nx.density(G),
        "transitivity": nx.transitivity(G) if m else np.nan,
        "average_clustering": nx.average_clustering(G) if m else np.nan,
        "components": nx.number_connected_components(G),
        "lcc_nodes": ln,
        "lcc_edges": lm,
        "lcc_share_nodes": ln / n if n else 0.0,
        "lcc_share_edges": lm / m if m else 0.0,
        "lcc_density": nx.density(L) if ln > 1 else 0.0,
        "lcc_transitivity": nx.transitivity(L) if lm else np.nan,
        "lcc_average_clustering": nx.average_clustering(L) if lm else np.nan,
    }



def graph_metrics_basic(G: nx.Graph) -> dict:
    """Fast metrics for sensitivity grids; excludes clustering/triangle scans."""
    n = G.number_of_nodes(); m = G.number_of_edges()
    if n == 0:
        return {"nodes":0,"edges":0,"mean_degree":0.0,"density":0.0,"components":0,"lcc_nodes":0,"lcc_edges":0,"lcc_share_nodes":0.0,"lcc_share_edges":0.0}
    L = largest_component_graph(G); ln=L.number_of_nodes(); lm=L.number_of_edges()
    return {
        "nodes":n,"edges":m,"mean_degree":2.0*m/n,"density":nx.density(G),
        "components":nx.number_connected_components(G),"lcc_nodes":ln,"lcc_edges":lm,
        "lcc_share_nodes":ln/n,"lcc_share_edges":lm/m if m else 0.0,
    }


def yearly_document_type_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for year,g in df.groupby("year"):
        total=len(g)
        counts={"Article":0,"Proceedings Paper":0,"Meeting Abstract":0,"Review":0,"Note":0,"Editorial Material":0,"Letter":0,"Book Chapter":0,"Retracted Publication":0}
        for types in g["document_types"]:
            for t in set(types or []):
                if t in counts: counts[t]+=1
        row={"year":int(year),"total_records":int(total)}
        for t,c in counts.items():
            key=t.lower().replace(" ","_")
            row[f"n_{key}"]=int(c); row[f"share_{key}"]=float(c/total) if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

def yearly_team_size_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("year"):
        sizes = g["team_size"].astype(int)
        rows.append({
            "year": int(year),
            "papers": int(len(g)),
            "mean_authors": float(sizes.mean()),
            "median_authors": float(sizes.median()),
            "p90_authors": float(sizes.quantile(0.90)),
            "p95_authors": float(sizes.quantile(0.95)),
            "max_authors": int(sizes.max()),
            "share_single_author": float((sizes == 1).mean()),
            "share_2_3_authors": float(sizes.between(2, 3).mean()),
            "share_4_9_authors": float(sizes.between(4, 9).mean()),
            "share_10_19_authors": float(sizes.between(10, 19).mean()),
            "share_20plus_authors": float((sizes >= 20).mean()),
            "share_50plus_authors": float((sizes >= 50).mean()),
            "share_100plus_authors": float((sizes >= 100).mean()),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def yearly_network_metrics(
    df: pd.DataFrame,
    start_year: int = 1960,
    end_year: int = 2007,
    document_type_mode: str = "all_selected",
    max_team_size: int | None = None,
    detailed: bool = True,
) -> pd.DataFrame:
    rows = []
    for year in range(start_year, end_year + 1):
        sub = filter_publications(
            df, year, year, max_team_size=max_team_size,
            document_type_mode=document_type_mode,
        )
        G = build_coauthorship_graph(sub)
        metrics = graph_metrics(G) if detailed else graph_metrics_basic(G)
        metrics.update({
            "year": year,
            "papers": int(len(sub)),
            "document_type_mode": document_type_mode,
            "max_team_size": max_team_size if max_team_size is not None else "none",
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def snapshot_metrics(
    df: pd.DataFrame,
    periods: Sequence[tuple[str, int, int]],
    team_thresholds: Sequence[int | None] = (None, 100, 50, 20),
    document_type_modes: Sequence[str] = ("all_selected",),
) -> pd.DataFrame:
    rows = []
    for label, start, end in periods:
        for mode in document_type_modes:
            for threshold in team_thresholds:
                sub = filter_publications(df, start, end, threshold, mode)
                G = build_coauthorship_graph(sub)
                row = graph_metrics_basic(G)
                row.update({
                    "period": label, "start_year": start, "end_year": end,
                    "papers": int(len(sub)), "document_type_mode": mode,
                    "max_team_size": threshold if threshold is not None else "none",
                })
                rows.append(row)
    return pd.DataFrame(rows)


def large_team_papers(df: pd.DataFrame, min_team_size: int = 20) -> pd.DataFrame:
    cols = [c for c in ["record_id", "ut", "year", "title_one_line", "source_title", "document_type_raw", "team_size"] if c in df.columns]
    out = df[df["team_size"] >= min_team_size][cols].copy()
    return out.sort_values(["year", "team_size"], ascending=[True, False]).reset_index(drop=True)


def _canonical_graph_order(G: nx.Graph) -> nx.Graph:
    """Return an insertion-order-canonical copy for stochastic graph routines.

    NetworkX Louvain uses a random seed but can still be sensitive to the input
    graph's internal insertion order. Rebuilding the graph with sorted nodes and
    edges makes a fixed seed reproducible across Python hash seeds/processes.
    """
    H=nx.Graph()
    for n,data in sorted(G.nodes(data=True),key=lambda x:str(x[0])):
        H.add_node(n,**dict(data))
    edges=[]
    for u,v,data in G.edges(data=True):
        a,b=sorted((u,v),key=str)
        edges.append((a,b,dict(data)))
    for u,v,data in sorted(edges,key=lambda x:(str(x[0]),str(x[1]))):
        H.add_edge(u,v,**data)
    return H


def louvain_partition(G: nx.Graph, seed: int = 42, weight: str = "paper_count", resolution: float = 1.0) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    H=_canonical_graph_order(G)
    if H.number_of_edges() == 0:
        return {n: i for i,n in enumerate(sorted(H.nodes(),key=str))}
    communities = nx.community.louvain_communities(H, weight=weight, seed=seed, resolution=resolution)
    partition = {}
    for cid, nodes in enumerate(sorted(communities, key=lambda x: (-len(x), min(map(str,x)) if x else ""))):
        for n in nodes:
            partition[n] = cid
    return partition


def louvain_partition_sensitivity(
    G: nx.Graph,
    *,
    reference_seed: int = 42,
    seeds: Sequence[int] = tuple(range(20)),
    weight: str = "paper_count",
    resolution: float = 1.0,
) -> pd.DataFrame:
    """Compare alternative Louvain partitions with seed 42 using ARI."""
    from sklearn.metrics import adjusted_rand_score

    nodes = sorted(G.nodes(), key=str)
    reference = louvain_partition(G, reference_seed, weight, resolution)
    rows = []
    for seed in seeds:
        partition = louvain_partition(G, int(seed), weight, resolution)
        labels_ref = [reference[n] for n in nodes]
        labels_alt = [partition[n] for n in nodes]
        communities = [
            {node for node, cid in partition.items() if cid == value}
            for value in sorted(set(partition.values()))
        ]
        modularity = (
            nx.community.modularity(G, communities, weight=weight, resolution=resolution)
            if G.number_of_edges() else np.nan
        )
        rows.append({
            "seed": int(seed),
            "reference_seed": int(reference_seed),
            "adjusted_rand_index": float(adjusted_rand_score(labels_ref, labels_alt)),
            "communities": len(communities),
            "modularity": float(modularity),
        })
    return pd.DataFrame(rows)


def centrality_table(G: nx.Graph, exact_betweenness_max_nodes: int = 1000, seed: int = 42) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["author", "degree", "weighted_degree", "fractional_degree", "betweenness", "betweenness_method"])
    n = G.number_of_nodes()
    if n <= exact_betweenness_max_nodes:
        bc = nx.betweenness_centrality(G, normalized=True, endpoints=False, weight=None)
        method = "exact_unweighted_normalized"
    else:
        k = min(250, n)
        bc = nx.betweenness_centrality(G, k=k, normalized=True, endpoints=False, weight=None, seed=seed)
        method = f"approx_unweighted_normalized_k{k}"
    part = louvain_partition(G, seed=seed, weight="paper_count")
    rows = []
    for a in G.nodes():
        rows.append({
            "author": a,
            "degree": int(G.degree(a)),
            "weighted_degree": float(G.degree(a, weight="weight")),
            "fractional_degree": float(G.degree(a, weight="fractional_weight")),
            "betweenness": float(bc.get(a, 0.0)),
            "community": int(part.get(a, -1)),
            "betweenness_method": method,
        })
    return pd.DataFrame(rows).sort_values(["betweenness", "degree"], ascending=False).reset_index(drop=True)


def period_network_bundle(df: pd.DataFrame, start: int, end: int, seed: int = 42) -> dict:
    sub = filter_publications(df, start, end)
    G = build_coauthorship_graph(sub, store_record_ids=False)
    L = largest_component_graph(G)
    return {
        "publications": sub,
        "graph": G,
        "lcc": L,
        "metrics": graph_metrics(G),
        "centrality_lcc": centrality_table(L, seed=seed),
    }


def author_overlap_1983_1986(df: pd.DataFrame) -> pd.DataFrame:
    s83 = filter_publications(df, 1983, 1983)
    s86 = filter_publications(df, 1986, 1986)
    g83 = build_coauthorship_graph(s83)
    g86 = build_coauthorship_graph(s86)
    l86 = largest_component_graph(g86)
    a83 = set(g83.nodes())
    a86l = set(l86.nodes())
    inter = a83 & a86l
    return pd.DataFrame([{
        "authors_1983": len(a83),
        "authors_1986_lcc": len(a86l),
        "authors_1983_reappearing_in_1986_lcc": len(inter),
        "share_of_1983_authors": len(inter) / len(a83) if a83 else np.nan,
        "share_of_1986_lcc": len(inter) / len(a86l) if a86l else np.nan,
    }])


def graph_to_graphml_safe(G: nx.Graph, path: str) -> None:
    """Write a GraphML-safe copy without altering the supplied graph.

    Kept as a compact compatibility wrapper. New diagnostic exports should use
    :mod:`fusion_pipeline.graph_export`, which also supports attribute profiles,
    publication provenance, title-term summaries, and compressed GraphMLZ.
    """
    from .graph_export import graphml_safe_copy
    nx.write_graphml(graphml_safe_copy(G), path)
