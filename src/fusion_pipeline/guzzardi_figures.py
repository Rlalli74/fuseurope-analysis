"""Reproducible network-figure helpers for the Guzzardi-Lalli paper.

The functions are paper-facing wrappers around the canonical co-authorship graph.
They keep analysis separate from display: full graph/LCC membership and Louvain
partitions are calculated from the complete graph, while betweenness can be
sampled only for node sizing/label selection in the legacy single-network views.

v0.13 adds a deliberately simpler four-panel overview figure.  Its purpose is
illustrative rather than inferential: it shows how a progressively larger share
of authors is absorbed into the largest connected component (LCC).  The
multipanel therefore does not encode Louvain communities, brokerage, or
betweenness centrality.
"""
from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from networkx.drawing.nx_agraph import graphviz_layout

from .networks import louvain_partition


LCC_COLOR = "#356a9a"
OTHER_COMPONENT_COLOR = "#c7c7c7"
EDGE_COLOR = "#8a8a8a"


def betweenness_for_display(G: nx.Graph, exact_max_nodes: int=1000, samples: int=500, seed: int=42):
    n=G.number_of_nodes()
    if n <= exact_max_nodes:
        return nx.betweenness_centrality(G, normalized=True, weight=None), "exact"
    k=min(int(samples),n)
    return nx.betweenness_centrality(G,k=k,normalized=True,weight=None,seed=seed), f"sampled_k{k}_seed{seed}"


def sfdp_layout(G: nx.Graph, seed: int=42, pack_components: bool=False):
    if G.number_of_nodes() == 0:
        return {}
    args=f"-Gstart={seed} -Goverlap=prism -Gsplines=false"
    if pack_components:
        args += " -Gpack=true -Gpackmode=graph"
    pos=graphviz_layout(G,prog="sfdp",args=args)
    a=np.array(list(pos.values()),dtype=float)
    mid=(a.min(0)+a.max(0))/2; span=max((a.max(0)-a.min(0)).max(),1e-12)
    return {n:(np.array(v)-mid)/span*2 for n,v in pos.items()}


def _scale_sizes(values,lo=3.0,hi=90.0):
    a=np.asarray(values,dtype=float)
    if not len(a): return a
    cap=np.quantile(a,.995) if len(a)>100 else a.max()
    a=np.minimum(a,cap)
    z=(a-a.min())/(a.max()-a.min()+1e-15)
    return lo+(hi-lo)*np.sqrt(z)


def _overview_node_size(n_nodes: int) -> float:
    """Same size rule for all panels, scaled only to network order.

    A modest n^-0.35 scaling keeps the 2001 panel legible without implying that
    node size represents an analytical variable.
    """
    if n_nodes <= 0:
        return 2.0
    return float(np.clip(7.0 * (1000.0 / n_nodes) ** 0.35, 2.3, 7.0))


def _overview_edge_alpha(n_edges: int) -> float:
    if n_edges <= 0:
        return 0.02
    return float(np.clip(1800.0 / n_edges, 0.010, 0.050))


def network_overview_stats(G: nx.Graph) -> dict:
    """Return the descriptive quantities printed in multipanel titles."""
    n_nodes=G.number_of_nodes()
    n_edges=G.number_of_edges()
    if n_nodes == 0:
        return {
            "nodes": 0,
            "edges": n_edges,
            "components": 0,
            "lcc_nodes": 0,
            "lcc_share": np.nan,
        }
    components=nx.number_connected_components(G)
    lcc_nodes=len(max(nx.connected_components(G),key=len))
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "components": components,
        "lcc_nodes": lcc_nodes,
        "lcc_share": lcc_nodes/n_nodes,
    }


def plot_full_network(G: nx.Graph, output: str|Path, *, seed: int=42, samples: int=500,
                      labels: int=0, width_inches: float=7.0, dpi: int=360):
    """Legacy single full-network view retained for repository/SI reproducibility."""
    b,method=betweenness_for_display(G,samples=samples,seed=seed)
    pos=sfdp_layout(G,seed=seed,pack_components=True)
    lcc=set(max(nx.connected_components(G),key=len)) if len(G) else set()
    nodes=list(G); sizes=_scale_sizes([b[n] for n in nodes],2.2,65)
    colors=[LCC_COLOR if n in lcc else OTHER_COMPONENT_COLOR for n in nodes]
    fig,ax=plt.subplots(figsize=(width_inches,width_inches*0.93))
    nx.draw_networkx_edges(G,pos,ax=ax,edge_color=EDGE_COLOR,alpha=.03,width=.12)
    nx.draw_networkx_nodes(G,pos,ax=ax,node_size=sizes,node_color=colors,linewidths=0,alpha=.83)
    top=[x for x,_ in sorted(b.items(),key=lambda kv:kv[1],reverse=True)[:int(labels)]]
    offsets=[(5,5),(5,-8),(-5,5),(-5,-8),(8,0),(-8,0),(5,10),(-5,10)]
    for j,n in enumerate(top):
        x,y=pos[n];dx,dy=offsets[j%len(offsets)]
        ax.annotate(n,(x,y),xytext=(dx,dy),textcoords="offset points",fontsize=5.2,
                    ha="left" if dx>=0 else "right",
                    bbox=dict(boxstyle="round,pad=.13",fc="white",ec=".76",lw=.3,alpha=.93))
    ax.axis("off");ax.margins(.02);fig.tight_layout(pad=.02)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=dpi,bbox_inches="tight",pad_inches=.03,facecolor="white");plt.close(fig)
    return {"betweenness_method":method,"label_nodes":top,"lcc_nodes":len(lcc)}


def plot_full_network_multipanel(
    graphs: Mapping[int, nx.Graph],
    output: str|Path,
    *,
    years: list[int] | tuple[int, ...] | None = None,
    seed: int = 42,
    width_inches: float = 9.0,
    dpi: int = 360,
    tiff_output: str|Path|None = None,
    tiff_dpi: int = 600,
):
    """Plot the four canonical full-network snapshots in one 2x2 figure.

    Visual encoding is intentionally minimal:
      * LCC nodes: blue
      * nodes in all other connected components: light grey
      * node size: display-only function of network order, identical rule per panel
      * no Louvain colours, author labels, or betweenness encoding

    Each panel reports author count, LCC share, and number of components.  These
    are descriptive statistics, not substitutes for the longitudinal network
    metrics reported elsewhere in the paper.
    """
    if years is None:
        years=sorted(graphs)
    years=list(years)
    if len(years) != 4:
        raise ValueError("The canonical Guzzardi-Lalli overview requires exactly four snapshot years.")
    missing=[y for y in years if y not in graphs]
    if missing:
        raise KeyError(f"Missing graph(s) for snapshot year(s): {missing}")

    fig,axes=plt.subplots(2,2,figsize=(width_inches,width_inches*0.82))
    axes=np.asarray(axes).ravel()
    audit=[]

    for i,(ax,year) in enumerate(zip(axes,years)):
        G=graphs[year]
        stats=network_overview_stats(G)
        lcc=set(max(nx.connected_components(G),key=len)) if len(G) else set()
        pos=sfdp_layout(G,seed=seed,pack_components=True)
        nodes=list(G)
        node_size=_overview_node_size(stats["nodes"])
        colors=[LCC_COLOR if n in lcc else OTHER_COMPONENT_COLOR for n in nodes]

        nx.draw_networkx_edges(
            G,pos,ax=ax,edge_color=EDGE_COLOR,
            alpha=_overview_edge_alpha(stats["edges"]),width=.10,
        )
        nx.draw_networkx_nodes(
            G,pos,ax=ax,node_size=node_size,node_color=colors,
            linewidths=0,alpha=.88,
        )
        panel=chr(ord("A")+i)
        share=100.0*stats["lcc_share"] if np.isfinite(stats["lcc_share"]) else float("nan")
        ax.set_title(
            f"{panel}  {year}\n"
            f"{stats['nodes']:,} authors · LCC {share:.1f}% · {stats['components']:,} components",
            fontsize=9.2,pad=5.0,
        )
        ax.axis("off");ax.margins(.025)
        audit.append({"year":year,**stats,"panel":panel,"node_size_display":node_size})

    legend_handles=[
        Patch(facecolor=LCC_COLOR,edgecolor="none",label="Largest connected component (LCC)"),
        Patch(facecolor=OTHER_COMPONENT_COLOR,edgecolor="none",label="Other connected components"),
    ]
    fig.legend(handles=legend_handles,loc="lower center",ncol=2,frameon=False,fontsize=8.5,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.015,right=.985,top=.965,bottom=.075,wspace=.025,hspace=.12)

    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=dpi,bbox_inches="tight",pad_inches=.04,facecolor="white")
    if tiff_output is not None:
        tiff_output=Path(tiff_output);tiff_output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(tiff_output,dpi=tiff_dpi,bbox_inches="tight",pad_inches=.04,facecolor="white",format="tiff")
    plt.close(fig)
    return pd.DataFrame(audit)


def plot_lcc_network(G: nx.Graph, output: str|Path, *, seed: int=42, samples: int=500,
                     labels: int=10, resolution: float=1.0, width_inches: float=7.0, dpi: int=360):
    """Legacy analytical LCC view retained for repository/SI reproducibility."""
    L=G.subgraph(max(nx.connected_components(G),key=len)).copy() if len(G) else nx.Graph()
    b,method=betweenness_for_display(L,samples=samples,seed=seed)
    pos=sfdp_layout(L,seed=seed,pack_components=False)
    part=louvain_partition(L,seed=seed,weight="paper_count",resolution=resolution)
    comm_ids=sorted(set(part.values()))
    comms=[{n for n,cid in part.items() if cid==c} for c in comm_ids]
    ci=[part[n] for n in L]; sizes=_scale_sizes([b[n] for n in L],2.8,88)
    fig,ax=plt.subplots(figsize=(width_inches,width_inches*0.93))
    nx.draw_networkx_edges(L,pos,ax=ax,edge_color="#777777",alpha=.024,width=.10)
    nx.draw_networkx_nodes(L,pos,ax=ax,node_size=sizes,node_color=ci,cmap=plt.get_cmap("turbo",len(comms)),vmin=-.5,vmax=len(comms)-.5,linewidths=0,alpha=.88)
    top=[x for x,_ in sorted(b.items(),key=lambda kv:kv[1],reverse=True)[:int(labels)]]
    offsets=[(5,5),(5,-8),(-5,5),(-5,-8),(8,0),(-8,0),(5,10),(-5,10),(10,-3),(-10,-3)]
    for j,n in enumerate(top):
        x,y=pos[n];dx,dy=offsets[j%len(offsets)]
        ax.annotate(n,(x,y),xytext=(dx,dy),textcoords="offset points",fontsize=5.0,
                    ha="left" if dx>=0 else "right",
                    bbox=dict(boxstyle="round,pad=.13",fc="white",ec=".72",lw=.3,alpha=.93))
    ax.axis("off");ax.margins(.03);fig.tight_layout(pad=.02)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=dpi,bbox_inches="tight",pad_inches=.03,facecolor="white");plt.close(fig)
    node_table=pd.DataFrame({"node":list(L),"betweenness":[b[n] for n in L],"community":[part[n] for n in L]})
    return {"betweenness_method":method,"label_nodes":top,"lcc_nodes":len(L),"communities":len(comms),"node_table":node_table}
