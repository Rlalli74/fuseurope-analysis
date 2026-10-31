"""Reusable, parameter-driven visualizations for canonical FusEurope networks."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import math

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from .exploration import compute_centralities, community_table
from .networks import largest_component_graph


def _resolve_output(path_or_dir, filename: str) -> Path:
    p=Path(path_or_dir)
    if p.suffix:
        p.parent.mkdir(parents=True,exist_ok=True); return p
    p.mkdir(parents=True,exist_ok=True); return p/filename


def plot_longitudinal_metric(
    metrics_df: pd.DataFrame,
    metric: str,
    plot_start_year: int | None = None,
    plot_end_year: int | None = None,
    x_col: str = "end_year",
    output: str | Path | None = None,
    show: bool = False,
):
    """Plot one selected longitudinal metric over a user-selected display range."""
    if metric not in metrics_df.columns: raise ValueError(f"Metric not found: {metric}")
    d=metrics_df.copy()
    if plot_start_year is not None: d=d[d[x_col]>=int(plot_start_year)]
    if plot_end_year is not None: d=d[d[x_col]<=int(plot_end_year)]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(d[x_col],d[metric],marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric.replace("_"," ").title())
    ax.set_title(f"{metric.replace('_',' ').title()} over time")
    ax.grid(True,alpha=.25)
    fig.tight_layout()
    if output is not None:
        p=_resolve_output(output,f"{metric}_over_time.png"); fig.savefig(p,dpi=300,bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return fig


def plot_longitudinal_metrics(
    metrics_df: pd.DataFrame,
    metrics: Sequence[str],
    plot_start_year: int | None = None,
    plot_end_year: int | None = None,
    output_dir: str | Path = "figures",
    show: bool = False,
) -> list[Path]:
    paths=[]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    for metric in metrics:
        path=out/f"{metric}_over_time.png"
        plot_longitudinal_metric(metrics_df,metric,plot_start_year,plot_end_year,output=path,show=show)
        paths.append(path)
    return paths


def plot_network(
    G: nx.Graph,
    scope: str = "lcc",
    node_size_by: str = "betweenness",
    color_by: str = "community",
    label_top_n: int = 20,
    min_degree: int = 0,
    max_nodes: int | None = None,
    layout: str = "spring",
    seed: int = 42,
    community_resolution: float = 1.0,
    betweenness_exact_max_nodes: int = 1000,
    betweenness_samples: int = 250,
    output: str | Path | None = None,
    show: bool = False,
):
    """Visualize a graph with user-selectable size metric and labeling.

    ``max_nodes`` is an explicit display filter only; it never changes the graph
    used for the analysis. Nodes are ranked by ``node_size_by`` when trimming.
    """
    scope=scope.lower(); H=largest_component_graph(G) if scope=="lcc" else G.copy()
    if min_degree>0:
        keep=[n for n,d in H.degree() if d>=min_degree]; H=H.subgraph(keep).copy()
    if H.number_of_nodes()==0: raise ValueError("No nodes remain for visualization")
    cent=compute_centralities(H,[node_size_by] if node_size_by!="community" else ["degree"],scope="full",seed=seed,betweenness_exact_max_nodes=betweenness_exact_max_nodes,betweenness_samples=betweenness_samples)
    size_measure=node_size_by if node_size_by in cent.columns else "degree"
    score=dict(zip(cent.node,cent[size_measure]))
    if max_nodes is not None and H.number_of_nodes()>int(max_nodes):
        keep=sorted(H.nodes(),key=lambda n:score.get(n,0),reverse=True)[:int(max_nodes)];H=H.subgraph(keep).copy()
        cent=cent[cent.node.isin(keep)];score={n:score[n] for n in keep}
    vals=np.array([max(float(score.get(n,0)),0.0) for n in H.nodes()],dtype=float)
    if np.all(vals==0): sizes=np.full(len(vals),40.0)
    else:
        scaled=(vals-vals.min())/(vals.max()-vals.min()+1e-12); sizes=25+325*np.sqrt(scaled)
    if color_by=="community":
        ct=community_table(H,scope="full",seed=seed,resolution=community_resolution); cmap=dict(zip(ct.node,ct.community)); colors=[cmap.get(n,0) for n in H.nodes()]
    else:
        colors=[0 for _ in H.nodes()]
    if layout=="spring": pos=nx.spring_layout(H,seed=seed,weight="weight")
    elif layout=="kamada_kawai": pos=nx.kamada_kawai_layout(H,weight="weight")
    elif layout=="circular": pos=nx.circular_layout(H)
    else: raise ValueError("layout must be spring, kamada_kawai, or circular")
    fig,ax=plt.subplots(figsize=(12,9))
    nx.draw_networkx_edges(H,pos,ax=ax,alpha=.15,width=.6)
    nx.draw_networkx_nodes(H,pos,ax=ax,node_size=sizes,node_color=colors,cmap=plt.get_cmap("viridis"),alpha=.85,linewidths=0)
    if label_top_n:
        labels_nodes=sorted(H.nodes(),key=lambda n:score.get(n,0),reverse=True)[:int(label_top_n)]
        nx.draw_networkx_labels(H,pos,labels={n:n for n in labels_nodes},font_size=7,ax=ax)
    ax.set_axis_off(); ax.set_title(f"Network ({scope}); node size = {node_size_by}")
    fig.tight_layout()
    if output is not None:
        p=_resolve_output(output,"network.png");fig.savefig(p,dpi=300,bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return fig
