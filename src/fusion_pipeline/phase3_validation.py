"""Validation and impact analysis for phase-3 author disambiguation."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence
import glob
import os

import networkx as nx
import pandas as pd

from .disambiguation import safe_text


DEFAULT_PERIODS = [
    (1959, 1968, "1959-1968"),
    (1969, 1973, "1969-1973"),
    (1974, 1978, "1974-1978"),
    (1979, 1979, "1979"),
    (1983, 1983, "1983"),
    (1986, 1986, "1986"),
    (1996, 1996, "1996"),
    (2001, 2001, "2001"),
]


def build_coauthor_graph(records: Sequence[Mapping], start: int, end: int, authors_col: str) -> nx.Graph:
    G = nx.Graph()
    for r in records:
        try: year = int(r.get("year"))
        except Exception: continue
        if not (start <= year <= end):
            continue
        authors = []
        seen = set()
        for x in r.get(authors_col, []) or []:
            a = safe_text(x).upper()
            if a and a not in seen:
                seen.add(a); authors.append(a)
        G.add_nodes_from(authors)
        for a, b in combinations(authors, 2):
            if G.has_edge(a, b): G[a][b]["weight"] += 1
            else: G.add_edge(a, b, weight=1)
    return G


def graph_metrics(G: nx.Graph) -> dict:
    n = G.number_of_nodes(); e = G.number_of_edges()
    if n == 0:
        return {"nodes": 0, "edges": 0, "components": 0, "lcc_size": 0, "lcc_share": 0.0, "mean_degree": 0.0}
    comps = list(nx.connected_components(G))
    lcc = max(comps, key=len) if comps else set()
    return {
        "nodes": n, "edges": e, "components": len(comps), "lcc_size": len(lcc),
        "lcc_share": len(lcc)/n if n else 0.0,
        "mean_degree": (2*e/n) if n else 0.0,
    }


def network_sensitivity(records: Sequence[Mapping], periods=DEFAULT_PERIODS) -> pd.DataFrame:
    rows=[]
    for start,end,label in periods:
        base=build_coauthor_graph(records,start,end,"authors_normalized")
        dis=build_coauthor_graph(records,start,end,"authors_network")
        mb=graph_metrics(base); md=graph_metrics(dis)
        row={"period":label,"start_year":start,"end_year":end}
        row.update({f"normalized_{k}":v for k,v in mb.items()})
        row.update({f"disambiguated_{k}":v for k,v in md.items()})
        row["delta_lcc_share"]=md["lcc_share"]-mb["lcc_share"]
        row["delta_nodes"]=md["nodes"]-mb["nodes"]
        rows.append(row)
    return pd.DataFrame(rows)


def add_review_impact(candidates: pd.DataFrame, records: Sequence[Mapping], periods=DEFAULT_PERIODS) -> pd.DataFrame:
    """Rank non-auto candidates by potential structural impact in manuscript periods.

    The calculation is diagnostic: for each baseline normalized network, it asks whether
    the two candidate names lie in different components. If so, merging them would connect
    those components. This does not assume that the merge is correct.
    """
    if candidates.empty:
        return candidates.copy()
    review=candidates[candidates["auto_merge"] != True].copy()  # noqa: E712
    if review.empty: return review

    period_info={}
    for start,end,label in periods:
        G=build_coauthor_graph(records,start,end,"authors_normalized")
        comp_id={}; comp_size={}; lcc_nodes=set()
        comps=list(nx.connected_components(G)) if G.number_of_nodes() else []
        if comps: lcc_nodes=max(comps,key=len)
        for i,c in enumerate(comps):
            for n in c: comp_id[n]=i; comp_size[n]=len(c)
        period_info[label]=(G,comp_id,comp_size,lcc_nodes)

    impact_rows=[]
    for _,r in review.iterrows():
        a,b=r["author_a"],r["author_b"]
        relevant=[]; connects=[]; touches_lcc=[]; max_join=0
        for label,(G,cid,csz,lcc) in period_info.items():
            if a in G and b in G:
                relevant.append(label)
                if cid.get(a)!=cid.get(b):
                    connects.append(label)
                    max_join=max(max_join,csz.get(a,0)+csz.get(b,0))
                    if a in lcc or b in lcc: touches_lcc.append(label)
        paper_mass=int(r.get("papers_a",0))+int(r.get("papers_b",0))
        priority_score=max_join*10 + paper_mass + int(r.get("shared_coauthors",0))*5 + int(r.get("shared_institutions",0))*3
        if touches_lcc or max_join >= 20 or paper_mass >= 20:
            priority="HIGH"
        elif connects or paper_mass >= 8 or int(r.get("shared_coauthors",0))>=1:
            priority="MEDIUM"
        else:
            priority="LOW"
        row=r.to_dict()
        row.update({
            "review_priority":priority,
            "review_priority_score":priority_score,
            "periods_both_present":" | ".join(relevant),
            "periods_connect_different_components":" | ".join(connects),
            "periods_touch_lcc_if_merged":" | ".join(touches_lcc),
            "max_potential_joined_component_size":max_join,
        })
        impact_rows.append(row)
    out=pd.DataFrame(impact_rows)
    rank={"HIGH":0,"MEDIUM":1,"LOW":2}
    out["_rank"]=out["review_priority"].map(rank)
    return out.sort_values(["_rank","review_priority_score","score"],ascending=[True,False,False]).drop(columns="_rank").reset_index(drop=True)


def disambiguation_summary(variant_map: pd.DataFrame, candidates: pd.DataFrame, merge_log: pd.DataFrame, review_impact: pd.DataFrame) -> dict:
    return {
        "distinct_normalized_author_forms": int(len(variant_map)),
        "variant_forms_merged": int(variant_map["was_merged"].sum()) if not variant_map.empty else 0,
        "canonical_author_nodes": int(variant_map["canonical_name"].nunique()) if not variant_map.empty else 0,
        "candidate_pairs_total": int(len(candidates)),
        "candidate_pairs_auto": int(candidates["auto_merge"].sum()) if not candidates.empty else 0,
        "candidate_pairs_review": int((candidates["auto_merge"] != True).sum()) if not candidates.empty else 0,
        "auto_merges_applied": int((merge_log["action"]=="auto_merge").sum()) if not merge_log.empty else 0,
        "auto_merges_blocked_transitive_conflict": int((merge_log["action"]=="blocked_transitive_conflict").sum()) if not merge_log.empty else 0,
        "review_high": int((review_impact["review_priority"]=="HIGH").sum()) if not review_impact.empty else 0,
        "review_medium": int((review_impact["review_priority"]=="MEDIUM").sum()) if not review_impact.empty else 0,
        "review_low": int((review_impact["review_priority"]=="LOW").sum()) if not review_impact.empty else 0,
    }


def compare_with_legacy_auto_merges(candidates: pd.DataFrame, legacy_root: str | Path | None):
    if not legacy_root or not Path(legacy_root).exists():
        return {}, pd.DataFrame()
    rows=[]
    for path in glob.glob(str(Path(legacy_root)/"**/*_author_auto_merges.csv"),recursive=True):
        if "__MACOSX" in path: continue
        try: d=pd.read_csv(path)
        except Exception: continue
        for _,r in d.iterrows():
            a=safe_text(r.get("Author_A") or r.get("author_a")).upper()
            b=safe_text(r.get("Author_B") or r.get("author_b")).upper()
            if a and b:
                x,y=sorted((a,b)); rows.append({"author_a":x,"author_b":y,"legacy_file":os.path.relpath(path,legacy_root)})
    legacy=pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["author_a","author_b","legacy_file"])
    if legacy.empty: return {"legacy_pairs":0},legacy
    new=candidates.copy()
    if not new.empty:
        new[["a_key","b_key"]]=new.apply(lambda r:pd.Series(sorted((r["author_a"],r["author_b"]))),axis=1)
        status={(r.a_key,r.b_key):("auto" if r.auto_merge else "review") for r in new.itertuples()}
    else: status={}
    legacy["new_status"]=[status.get((r.author_a,r.author_b),"not_candidate") for r in legacy.itertuples()]
    return {
        "legacy_pairs":int(len(legacy)),
        "legacy_pairs_new_auto":int((legacy["new_status"]=="auto").sum()),
        "legacy_pairs_new_review":int((legacy["new_status"]=="review").sum()),
        "legacy_pairs_not_candidate":int((legacy["new_status"]=="not_candidate").sum()),
    },legacy


def historical_manual_review_shortlist(
    candidates: pd.DataFrame,
    records: Sequence[Mapping],
    max_recommended: int = 30,
    manual_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a deliberately small manual-review shortlist for the Lalli historical paper.

    The many unresolved candidate pairs in the full 1956-2007 corpus are *not* a manual
    work queue. They remain separate by default. This function selects only ambiguous
    pairs that are both present in one of the historical network windows and whose merger
    could plausibly alter component structure or a relatively high-degree node.

    The top ``max_recommended`` rows are marked ``recommended_for_manual_check=1``;
    remaining rows are retained for transparency but need not be checked for the revision.
    """
    hist_periods = [
        (1959, 1968, "1959-1968"), (1969, 1973, "1969-1973"),
        (1974, 1978, "1974-1978"), (1983, 1983, "1983"), (1986, 1986, "1986"),
    ]
    if candidates.empty:
        return pd.DataFrame()
    review = candidates[candidates["auto_merge"] != True].copy()  # noqa: E712

    # Pairs already adjudicated in the curated override table are not a remaining
    # manual-review queue.  They remain visible in the override file and merge log.
    if manual_overrides is not None and not manual_overrides.empty and not review.empty:
        decided = set()
        for _, ov in manual_overrides.iterrows():
            a = safe_text(ov.get("author_a")).upper()
            b = safe_text(ov.get("author_b")).upper()
            dec = safe_text(ov.get("decision")).upper()
            if a and b and dec in {"MERGE", "YES", "SEPARATE", "KEEP_SEPARATE", "NO"}:
                decided.add(tuple(sorted((a, b))))
        if decided:
            mask = review.apply(
                lambda r: tuple(sorted((safe_text(r["author_a"]).upper(), safe_text(r["author_b"]).upper()))) not in decided,
                axis=1,
            )
            review = review[mask].copy()
    if review.empty:
        return review

    infos = {}
    for start, end, label in hist_periods:
        G = build_coauthor_graph(records, start, end, "authors_network")
        comps = list(nx.connected_components(G)) if G.number_of_nodes() else []
        lcc = max(comps, key=len) if comps else set()
        cid, csz = {}, {}
        for i, comp in enumerate(comps):
            for n in comp:
                cid[n] = i; csz[n] = len(comp)
        infos[label] = (G, cid, csz, lcc)

    rows = []
    for _, r in review.iterrows():
        a, b = r["author_a"], r["author_b"]
        both, connects, touches = [], [], []
        max_lcc_gain = 0
        max_degree = 0
        for label, (G, cid, csz, lcc) in infos.items():
            if a not in G or b not in G:
                continue
            both.append(label)
            max_degree = max(max_degree, int(G.degree(a)), int(G.degree(b)))
            if cid.get(a) != cid.get(b):
                connects.append(label)
                if a in lcc and b not in lcc:
                    max_lcc_gain = max(max_lcc_gain, csz.get(b, 0))
                    touches.append(label)
                elif b in lcc and a not in lcc:
                    max_lcc_gain = max(max_lcc_gain, csz.get(a, 0))
                    touches.append(label)
        if not both:
            continue

        imode = r.get("initials_mode")
        common = bool(r.get("common_surname_prefix_risk", 0))
        shared_co = int(r.get("shared_coauthors", 0))
        shared_inst = int(r.get("shared_institutions", 0))
        score = float(r.get("score", 0))

        # Identity plausibility filter: exact-initial surname variants are retained;
        # initial-prefix variants need extra caution for common surnames; conflicting
        # initials are considered only with unusually strong independent context.
        plausible = (
            imode == "exact"
            or (imode == "prefix" and not common)
            or (imode == "prefix" and common and shared_co >= 2 and shared_inst >= 1)
            or (imode == "conflict" and shared_co >= 2 and shared_inst >= 1 and score >= 95)
        )
        structural = (max_lcc_gain >= 2) or (bool(connects) and max_degree >= 10) or (max_degree >= 25)
        if not (plausible and structural):
            continue

        d = r.to_dict()
        d.update({
            "historical_periods_both_present": " | ".join(both),
            "historical_periods_connect_components": " | ".join(connects),
            "historical_periods_touch_lcc": " | ".join(touches),
            "max_lcc_gain_if_same_person": max_lcc_gain,
            "max_degree_in_historical_periods": max_degree,
        })
        rows.append(d)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(
        ["max_lcc_gain_if_same_person", "max_degree_in_historical_periods", "shared_coauthors", "score"],
        ascending=False,
    ).reset_index(drop=True)
    out.insert(0, "manual_review_rank", range(1, len(out) + 1))
    # A row can remain visible for transparency without becoming a manual work item.
    # For the historical-paper revision, recommend checking only unresolved identities
    # whose forced merger could add at least two nodes to the current LCC. High-degree
    # same-component ambiguities are handled by sensitivity analysis rather than by
    # requiring exhaustive hand-disambiguation.
    required = (out["max_lcc_gain_if_same_person"] >= 2)
    if max_recommended is not None:
        required &= (out["manual_review_rank"] <= max_recommended)
    out.insert(1, "recommended_for_manual_check", required.astype(int))
    return out
