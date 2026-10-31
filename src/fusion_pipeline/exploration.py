"""General diagnostic network analysis utilities.

This module preserves broader diagnostic analyses developed during the
FusEurope project while making them parameter-driven. Nothing here
is tied to the periods used in either paper: callers choose the analysis range,
time-window mode, metrics, centralities, and graph scope.

The same functions are used by the command-line scripts and can also support optional local notebooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
import math

import numpy as np
import pandas as pd
import networkx as nx

from .networks import (
    build_coauthorship_graph,
    filter_publications,
    largest_component_graph,
    louvain_partition,
)


AVAILABLE_GRAPH_METRICS = (
    "number_of_nodes",
    "number_of_edges",
    "average_degree",
    "density",
    "number_of_connected_components",
    "largest_component_size",
    "largest_component_edges",
    "ratio_largest_component_nodes",
    "ratio_largest_component_edges",
    "average_clustering_coefficient",
    "transitivity",
    "diameter_of_largest_component",
    "normalized_diameter_total_nodes",
    "normalized_diameter_largest_component_nodes",
    "average_shortest_path_length_lcc",
    "degree_assortativity",
    "modularity",
)

BASIC_GRAPH_METRICS = (
    "number_of_nodes",
    "number_of_edges",
    "average_degree",
    "density",
    "number_of_connected_components",
    "largest_component_size",
    "largest_component_edges",
    "ratio_largest_component_nodes",
    "ratio_largest_component_edges",
)

EXTENDED_GRAPH_METRICS = AVAILABLE_GRAPH_METRICS

AVAILABLE_CENTRALITIES = (
    "degree",
    "degree_normalized",
    "weighted_degree",
    "fractional_degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "eccentricity",
)

AVAILABLE_ANALYSES = (
    "metrics",
    "centralities",
    "communities",
    "bridge_scores",
    "graph_export",
)


@dataclass(frozen=True)
class TimeWindow:
    label: str
    start_year: int
    end_year: int


def make_time_windows(
    start_year: int,
    end_year: int,
    mode: str = "annual",
    window_size: int = 1,
    step: int = 1,
) -> list[TimeWindow]:
    """Create user-selectable analysis windows.

    Parameters
    ----------
    mode:
        ``annual`` creates one graph per year; ``rolling`` creates fixed-width
        rolling windows ending at successive years; ``period`` creates one graph
        for the complete start/end range.
    """
    start_year = int(start_year); end_year = int(end_year)
    window_size = int(window_size); step = int(step)
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if window_size < 1 or step < 1:
        raise ValueError("window_size and step must be >= 1")
    mode = str(mode).lower()
    if mode == "annual":
        return [TimeWindow(str(y), y, y) for y in range(start_year, end_year + 1, step)]
    if mode == "period":
        return [TimeWindow(f"{start_year}-{end_year}", start_year, end_year)]
    if mode == "rolling":
        first_end = start_year + window_size - 1
        if first_end > end_year:
            raise ValueError("rolling window_size is larger than the selected analysis range")
        return [
            TimeWindow(f"{e-window_size+1}-{e}", e-window_size+1, e)
            for e in range(first_end, end_year + 1, step)
        ]
    raise ValueError("mode must be one of: annual, rolling, period")


def _safe_lcc(G: nx.Graph) -> nx.Graph:
    return largest_component_graph(G) if G.number_of_nodes() else nx.Graph()


def _diameter(G: nx.Graph, exact_max_nodes: int = 1000, seed: int = 42) -> tuple[float, str]:
    if G.number_of_nodes() == 0:
        return np.nan, "not_applicable"
    if G.number_of_nodes() == 1:
        return 0.0, "exact"
    if G.number_of_nodes() <= exact_max_nodes:
        return float(nx.diameter(G)), "exact"
    try:
        from networkx.algorithms import approximation as approx
        return float(approx.diameter(G, seed=seed)), "approximation"
    except TypeError:
        from networkx.algorithms import approximation as approx
        return float(approx.diameter(G)), "approximation"


def _modularity(G: nx.Graph, method: str = "louvain", seed: int = 42, weight: str | None = "weight") -> tuple[float, int]:
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 0.0, G.number_of_nodes()
    method = method.lower()
    if method == "louvain":
        part = louvain_partition(G, seed=seed, weight=weight)
        by_c = {}
        for node, cid in part.items():
            by_c.setdefault(cid, set()).add(node)
        comms = [by_c[cid] for cid in sorted(by_c)]
    elif method == "greedy":
        # Kept to reproduce earlier longitudinal diagnostic runs.
        comms = nx.community.greedy_modularity_communities(G, weight=weight)
    else:
        raise ValueError("modularity_method must be 'louvain' or 'greedy'")
    return float(nx.community.modularity(G, comms, weight=weight)), len(comms)


def compute_graph_metrics(
    G: nx.Graph,
    metrics: Sequence[str] = BASIC_GRAPH_METRICS,
    modularity_method: str = "louvain",
    modularity_weight: str | None = "weight",
    diameter_exact_max_nodes: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute only the graph metrics explicitly requested by the user."""
    requested = list(dict.fromkeys(metrics))
    unknown = sorted(set(requested) - set(AVAILABLE_GRAPH_METRICS))
    if unknown:
        raise ValueError(f"Unknown graph metrics: {unknown}")

    n = G.number_of_nodes(); m = G.number_of_edges()
    L = None
    out: dict[str, object] = {}

    if "number_of_nodes" in requested: out["number_of_nodes"] = n
    if "number_of_edges" in requested: out["number_of_edges"] = m
    if "average_degree" in requested: out["average_degree"] = (2.0*m/n) if n else 0.0
    if "density" in requested: out["density"] = nx.density(G) if n > 1 else 0.0
    if "number_of_connected_components" in requested:
        out["number_of_connected_components"] = nx.number_connected_components(G) if n else 0
    if "average_clustering_coefficient" in requested:
        out["average_clustering_coefficient"] = nx.average_clustering(G) if m else 0.0
    if "transitivity" in requested:
        out["transitivity"] = nx.transitivity(G) if m else 0.0
    if "degree_assortativity" in requested:
        if n > 1 and m:
            with np.errstate(all="ignore"):
                try: out["degree_assortativity"] = float(nx.degree_assortativity_coefficient(G))
                except Exception: out["degree_assortativity"] = np.nan
        else:
            out["degree_assortativity"] = np.nan
    if "modularity" in requested:
        mod, ncom = _modularity(G, modularity_method, seed, modularity_weight)
        out["modularity"] = mod
        out["modularity_method"] = modularity_method
        out["modularity_n_communities"] = int(ncom)

    lcc_needed = bool(set(requested) & {
        "largest_component_size", "largest_component_edges",
        "ratio_largest_component_nodes", "ratio_largest_component_edges",
        "diameter_of_largest_component", "normalized_diameter_total_nodes",
        "normalized_diameter_largest_component_nodes", "average_shortest_path_length_lcc",
    })
    if lcc_needed:
        L = _safe_lcc(G)
        ln = L.number_of_nodes(); lm = L.number_of_edges()
        if "largest_component_size" in requested: out["largest_component_size"] = ln
        if "largest_component_edges" in requested: out["largest_component_edges"] = lm
        if "ratio_largest_component_nodes" in requested: out["ratio_largest_component_nodes"] = ln/n if n else 0.0
        if "ratio_largest_component_edges" in requested: out["ratio_largest_component_edges"] = lm/m if m else 0.0
        if set(requested) & {
            "diameter_of_largest_component", "normalized_diameter_total_nodes",
            "normalized_diameter_largest_component_nodes"
        }:
            diam, method = _diameter(L, diameter_exact_max_nodes, seed)
            if "diameter_of_largest_component" in requested: out["diameter_of_largest_component"] = diam
            out["diameter_method"] = method
            if "normalized_diameter_total_nodes" in requested:
                out["normalized_diameter_total_nodes"] = diam/n if n and np.isfinite(diam) else np.nan
            if "normalized_diameter_largest_component_nodes" in requested:
                out["normalized_diameter_largest_component_nodes"] = diam/ln if ln and np.isfinite(diam) else np.nan
        if "average_shortest_path_length_lcc" in requested:
            out["average_shortest_path_length_lcc"] = (
                float(nx.average_shortest_path_length(L)) if ln > 1 else (0.0 if ln == 1 else np.nan)
            )
    return out


def _componentwise_eccentricity(G: nx.Graph) -> dict:
    result = {}
    for nodes in nx.connected_components(G):
        H = G.subgraph(nodes)
        if H.number_of_nodes() == 1:
            node = next(iter(nodes)); result[node] = 0.0
        else:
            result.update({k: float(v) for k, v in nx.eccentricity(H).items()})
    return result


def _componentwise_eigenvector(G: nx.Graph, weight: str | None = None, max_iter: int = 1000) -> dict:
    result = {}
    for nodes in nx.connected_components(G):
        H = G.subgraph(nodes)
        if H.number_of_nodes() == 1:
            node = next(iter(nodes)); result[node] = 1.0
            continue
        try:
            vals = nx.eigenvector_centrality(H, weight=weight, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            vals = nx.eigenvector_centrality_numpy(H, weight=weight)
        result.update({k: float(v) for k, v in vals.items()})
    return result


def compute_centralities(
    G: nx.Graph,
    measures: Sequence[str] = ("degree", "betweenness"),
    scope: str = "lcc",
    betweenness_exact_max_nodes: int = 1000,
    betweenness_samples: int = 250,
    eigenvector_weight: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute a user-selected set of centrality measures.

    ``scope='lcc'`` reproduces the usual historical-figure logic.  ``scope='full'``
    is also supported; eccentricity and eigenvector are then calculated component
    by component so disconnected graphs do not fail.
    """
    requested = list(dict.fromkeys(measures))
    unknown = sorted(set(requested) - set(AVAILABLE_CENTRALITIES))
    if unknown: raise ValueError(f"Unknown centralities: {unknown}")
    scope = scope.lower()
    if scope not in {"lcc", "full"}: raise ValueError("scope must be 'lcc' or 'full'")
    H = _safe_lcc(G) if scope == "lcc" else G.copy()
    if H.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", *requested, "centrality_scope"])

    values: dict[str, Mapping] = {}
    methods: dict[str, str] = {}
    n = H.number_of_nodes()
    if "degree" in requested: values["degree"] = dict(H.degree())
    if "degree_normalized" in requested: values["degree_normalized"] = nx.degree_centrality(H)
    if "weighted_degree" in requested: values["weighted_degree"] = dict(H.degree(weight="weight"))
    if "fractional_degree" in requested:
        values["fractional_degree"] = dict(H.degree(weight="fractional_weight"))
    if "closeness" in requested: values["closeness"] = nx.closeness_centrality(H)
    if "eccentricity" in requested: values["eccentricity"] = _componentwise_eccentricity(H)
    if "eigenvector" in requested: values["eigenvector"] = _componentwise_eigenvector(H, eigenvector_weight)
    if "betweenness" in requested:
        if n <= betweenness_exact_max_nodes:
            values["betweenness"] = nx.betweenness_centrality(H, normalized=True, endpoints=False, weight=None)
            methods["betweenness"] = "exact_unweighted_normalized"
        else:
            k = min(int(betweenness_samples), n)
            values["betweenness"] = nx.betweenness_centrality(H, k=k, normalized=True, endpoints=False, weight=None, seed=seed)
            methods["betweenness"] = f"approx_unweighted_normalized_k{k}"

    rows=[]
    for node in H.nodes():
        row={"node":node,"centrality_scope":scope}
        for measure in requested:
            row[measure]=float(values[measure].get(node,0.0))
        rows.append(row)
    out=pd.DataFrame(rows)
    if "betweenness" in requested: out["betweenness_method"] = methods["betweenness"]
    sort_col = "betweenness" if "betweenness" in requested else requested[0]
    return out.sort_values(sort_col,ascending=False).reset_index(drop=True)


def community_table(G: nx.Graph, scope: str = "lcc", seed: int = 42, weight: str = "weight", resolution: float = 1.0) -> pd.DataFrame:
    scope=scope.lower(); H=_safe_lcc(G) if scope=="lcc" else G.copy()
    if H.number_of_nodes()==0: return pd.DataFrame(columns=["node","community","community_size","scope"])
    if H.number_of_edges()==0:
        part={n:i for i,n in enumerate(sorted(H.nodes(), key=str))}
    else:
        part=louvain_partition(H,seed=seed,weight=weight,resolution=resolution)
    sizes=pd.Series(part).value_counts().to_dict()
    return pd.DataFrame([{"node":n,"community":int(c),"community_size":int(sizes[c]),"scope":scope} for n,c in part.items()])


def compute_bridge_scores(
    G: nx.Graph,
    partition: Mapping[str,int] | None = None,
    scope: str = "lcc",
    betweenness_samples: int = 250,
    seed: int = 42,
    betweenness_exact_max_nodes: int = 1000,
) -> pd.DataFrame:
    """Diagnostic bridge score = betweenness × cross-community tie share."""
    H=_safe_lcc(G) if scope=="lcc" else G.copy()
    if H.number_of_nodes()==0:
        return pd.DataFrame(columns=["node","community","betweenness","external_links","cross_community_share","bridge_score"])
    if partition is None:
        ct=community_table(H,scope="full",seed=seed)
        partition=dict(zip(ct.node,ct.community))
    n=H.number_of_nodes(); k=min(betweenness_samples,n)
    bc=(nx.betweenness_centrality(H,normalized=True,endpoints=False,weight=None)
        if n<=betweenness_exact_max_nodes else nx.betweenness_centrality(H,k=k,normalized=True,endpoints=False,weight=None,seed=seed))
    rows=[]
    for node in H.nodes():
        deg=H.degree(node)
        ext=sum(1 for nbr in H.neighbors(node) if partition.get(nbr)!=partition.get(node)) if deg else 0
        share=ext/deg if deg else 0.0
        rows.append({"node":node,"community":partition.get(node),"betweenness":float(bc.get(node,0)),
                     "external_links":int(ext),"cross_community_share":float(share),
                     "bridge_score":float(bc.get(node,0))*share})
    return pd.DataFrame(rows).sort_values("bridge_score",ascending=False).reset_index(drop=True)


def longitudinal_author_metrics(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    mode: str = "annual",
    window_size: int = 1,
    step: int = 1,
    metrics: Sequence[str] = BASIC_GRAPH_METRICS,
    document_type_mode: str = "all_selected",
    max_team_size: int | None = None,
    modularity_method: str = "louvain",
    seed: int = 42,
) -> pd.DataFrame:
    rows=[]
    for w in make_time_windows(start_year,end_year,mode,window_size,step):
        sub=filter_publications(df,w.start_year,w.end_year,max_team_size,document_type_mode)
        G=build_coauthorship_graph(sub)
        row=compute_graph_metrics(G,metrics,modularity_method=modularity_method,seed=seed)
        row.update({"label":w.label,"start_year":w.start_year,"end_year":w.end_year,"papers":int(len(sub)),
                    "window_mode":mode,"window_size":window_size,"document_type_mode":document_type_mode,
                    "max_team_size":max_team_size if max_team_size is not None else "none"})
        rows.append(row)
    return pd.DataFrame(rows)


def centrality_over_time(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    measures: Sequence[str],
    nodes: Sequence[str] | None = None,
    top_n: int | None = None,
    mode: str = "annual",
    window_size: int = 1,
    step: int = 1,
    scope: str = "lcc",
    document_type_mode: str = "all_selected",
    max_team_size: int | None = None,
    seed: int = 42,
    betweenness_exact_max_nodes: int = 1000,
    betweenness_samples: int = 250,
) -> pd.DataFrame:
    """Centrality profiles across arbitrary annual/rolling windows.

    With ``nodes`` supplied, returns those focal authors even when their value is
    zero/missing in a window.  Otherwise returns all nodes or the top ``top_n``
    nodes per requested measure and window.
    """
    rows=[]
    focal=list(nodes or [])
    for w in make_time_windows(start_year,end_year,mode,window_size,step):
        sub=filter_publications(df,w.start_year,w.end_year,max_team_size,document_type_mode)
        G=build_coauthorship_graph(sub)
        cent=compute_centralities(G,measures,scope=scope,seed=seed,betweenness_exact_max_nodes=betweenness_exact_max_nodes,betweenness_samples=betweenness_samples)
        if focal:
            cent=cent.set_index("node")
            for node in focal:
                row={"node":node,"label":w.label,"start_year":w.start_year,"end_year":w.end_year,"centrality_scope":scope}
                for m in measures: row[m]=float(cent.loc[node,m]) if node in cent.index else 0.0
                rows.append(row)
        elif top_n:
            for measure in measures:
                tmp=cent.nlargest(int(top_n),measure)[["node",measure]].copy()
                for rank,(_,r) in enumerate(tmp.iterrows(),1):
                    rows.append({"node":r.node,"measure":measure,"value":float(r[measure]),"rank":rank,
                                 "label":w.label,"start_year":w.start_year,"end_year":w.end_year,"centrality_scope":scope})
        else:
            cent=cent.copy();cent["label"]=w.label;cent["start_year"]=w.start_year;cent["end_year"]=w.end_year
            rows.extend(cent.to_dict("records"))
    return pd.DataFrame(rows)


def analyze_author_period(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    analyses: Sequence[str] = ("metrics",),
    metrics: Sequence[str] = BASIC_GRAPH_METRICS,
    centralities: Sequence[str] = ("degree", "betweenness"),
    scope: str = "lcc",
    document_type_mode: str = "all_selected",
    max_team_size: int | None = None,
    modularity_method: str = "louvain",
    community_resolution: float = 1.0,
    seed: int = 42,
    betweenness_exact_max_nodes: int = 1000,
    betweenness_samples: int = 250,
) -> dict[str, object]:
    """One-stop dispatcher for a user-selected author-network period."""
    requested=list(dict.fromkeys(analyses)); unknown=sorted(set(requested)-set(AVAILABLE_ANALYSES))
    if unknown: raise ValueError(f"Unknown analyses: {unknown}")
    sub=filter_publications(df,start_year,end_year,max_team_size,document_type_mode)
    G=build_coauthorship_graph(sub,store_record_ids=False)
    out:dict[str,object]={"publications":sub,"graph":G}
    if "metrics" in requested:
        out["metrics"]=compute_graph_metrics(G,metrics,modularity_method=modularity_method,seed=seed)
    if "centralities" in requested:
        out["centralities"]=compute_centralities(G,centralities,scope=scope,seed=seed,betweenness_exact_max_nodes=betweenness_exact_max_nodes,betweenness_samples=betweenness_samples)
    if "communities" in requested or "bridge_scores" in requested:
        ct=community_table(G,scope=scope,seed=seed,resolution=community_resolution)
        out["communities"]=ct
        if "bridge_scores" in requested:
            part=dict(zip(ct.node,ct.community))
            out["bridge_scores"]=compute_bridge_scores(G,part,scope=scope,seed=seed,betweenness_samples=betweenness_samples,betweenness_exact_max_nodes=betweenness_exact_max_nodes)
    if "graph_export" in requested:
        out["lcc"]=_safe_lcc(G)
    return out


def parse_period_specs(specs: Sequence[str]) -> list[tuple[str,int,int]]:
    """Parse CLI/Jupyter period specs.

    Accepted forms: ``1986``, ``1974:1978``, or ``JET-era=1979:1986``.
    Returns tuples ``(label,start,end)``.
    """
    out=[]
    for raw in specs:
        text=str(raw).strip()
        label=None
        if '=' in text:
            label,text=text.split('=',1);label=label.strip()
        if ':' in text:
            a,b=text.split(':',1);start=int(a);end=int(b)
        else:
            start=end=int(text)
        if start>end: raise ValueError(f"Invalid period {raw}: start > end")
        out.append((label or (str(start) if start==end else f"{start}-{end}"),start,end))
    return out
