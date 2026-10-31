"""Validation helpers for author/country normalization (pipeline phase 2)."""
from __future__ import annotations
import ast, json
from pathlib import Path
from typing import List, Sequence
import pandas as pd


def _list_cell(v):
    if isinstance(v, list): return [str(x) for x in v]
    if pd.isna(v): return []
    try:
        x=ast.literal_eval(str(v))
        return [str(y) for y in x] if isinstance(x,list) else []
    except Exception: return []


def validate_author_count_invariance(records: Sequence[dict]) -> dict:
    n_bad=0; examples=[]
    for r in records:
        raw=r.get('authors_wos',[]) or []; norm=r.get('authors_normalized',[]) or []
        if len(raw)!=len(norm):
            n_bad+=1
            if len(examples)<20: examples.append({'record_id':r.get('record_id'),'raw':raw,'normalized':norm})
    return {'n_records':len(records),'records_with_changed_author_count':n_bad,'examples':examples}


def summarize_truncation_audit(audit: pd.DataFrame) -> dict:
    c=audit['status'].value_counts().to_dict() if not audit.empty else {}
    return {
        'n_unique_dot_truncated_forms': int(len(audit)),
        'status_counts': {str(k):int(v) for k,v in c.items()},
        'occurrences_auto_recovered': int(audit.loc[audit.status=='auto_unique_prefix','occurrences'].sum()) if not audit.empty else 0,
        'occurrences_unresolved_or_review': int(audit.loc[audit.status!='auto_unique_prefix','occurrences'].sum()) if not audit.empty else 0,
    }


def summarize_affiliations(aff: pd.DataFrame) -> dict:
    if aff.empty: return {}
    return {
        'n_affiliation_rows': int(len(aff)),
        'n_with_normalized_country': int(aff.country_normalized.notna().sum()),
        'country_coverage': float(aff.country_normalized.notna().mean()),
        'n_jet_affiliations': int(aff.is_jet_affiliation.sum()),
        'n_unique_records_with_jet': int(aff.loc[aff.is_jet_affiliation,'record_id'].nunique()),
        'normalization_method_counts': {str(k):int(v) for k,v in aff.country_normalization_method.value_counts().to_dict().items()},
    }


def compare_country_lists_to_historical(records: Sequence[dict], historical_csv: str|Path) -> tuple[dict,pd.DataFrame]:
    """Compare paper-level lists using (year,title); diagnostic, not a gold standard.

    Historical country lists contain manual/enriched information and the legacy JET
    transformation. Therefore discrepancies are expected and are preserved for audit.
    """
    old=pd.read_csv(historical_csv).copy()
    old['countries_old']=old['countries'].apply(_list_cell)
    # title representation in parsed records was designed to reproduce historical CSV.
    new=pd.DataFrame([{
        'record_id':r.get('record_id'),'year':r.get('year'),'title':r.get('title'),
        'countries_normalized':r.get('countries_normalized',[]),
        'countries_network':r.get('countries_network',[]),
    } for r in records])
    # duplicate year/title records are rare; use cumcount deterministically for diagnostics.
    new['_n']=new.groupby(['year','title'],dropna=False).cumcount()
    old['_n']=old.groupby(['year','title'],dropna=False).cumcount()
    m=new.merge(old[['year','title','_n','countries_old']],on=['year','title','_n'],how='outer',indicator=True)
    def seteq(a,b): return set(a if isinstance(a,list) else [])==set(b if isinstance(b,list) else [])
    m['normalized_set_match']=m.apply(lambda r:seteq(r.get('countries_normalized'),r.get('countries_old')),axis=1)
    m['network_set_match']=m.apply(lambda r:seteq(r.get('countries_network'),r.get('countries_old')),axis=1)
    summary={
        'matched_rows':int((m['_merge']=='both').sum()),
        'parsed_only':int((m['_merge']=='left_only').sum()),
        'historical_only':int((m['_merge']=='right_only').sum()),
        'normalized_country_set_match_rate_on_matched':float(m.loc[m['_merge']=='both','normalized_set_match'].mean()),
        'network_country_set_match_rate_on_matched':float(m.loc[m['_merge']=='both','network_set_match'].mean()),
    }
    return summary,m
