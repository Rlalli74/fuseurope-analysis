"""General period-contrast utility consolidated from earlier textual trend scripts.

Earlier diagnostic notebooks compared hard-coded periods (notably 1968–74
and 1993–96) using document-frequency log-odds for terms, authors,
institutions, and countries.  This module retains that analytical capability but
makes both focal/baseline periods and feature families user-selectable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence
import ast, math, re, unicodedata

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .internationalization import parse_list_cell

DEFAULT_STOPWORDS = {
    "the","a","an","and","or","of","in","on","for","to","from","with","without",
    "by","at","into","over","under","between","among","through","during","after",
    "before","about","against","within","is","are","was","were","be","been","being",
    "this","that","these","those","we","our","their","it","its","as","such",
    "study","studies","analysis","report","reports","paper","work","results","new",
    "approach","method","methods","using","used","use","based","effect","effects",
    "model","models","system","systems","investigation","observations","observation",
}

FINAL_HISTORICAL_CONTRASTS = (
    ("1968-1974_vs_1956-1967", 1968, 1974, 1956, 1967),
    ("1994-1996_vs_1989-1993", 1994, 1996, 1989, 1993),
)
FINAL_HISTORICAL_TOKEN_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)


def final_historical_title_features(title: object) -> set[str]:
    """Frozen contextual title features from the final Guzzardi–Lalli SI.

    This deliberately does not reuse the community-titleword stopword or merge
    rules: the final contextual procedure is lower-case unigram plus bigram,
    removes punctuation while retaining internal hyphens, and counts document
    presence.  Returning a set enforces the latter rule.
    """
    text = unicodedata.normalize("NFKC", "" if title is None else str(title)).lower()
    tokens = FINAL_HISTORICAL_TOKEN_RE.findall(text)
    return set(tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])])


def final_historical_title_contrast(
    df: pd.DataFrame,
    focal_start: int,
    focal_end: int,
    baseline_start: int,
    baseline_end: int,
    *,
    min_document_frequency: int = 3,
    correction: float = 0.5,
) -> pd.DataFrame:
    """Run one frozen historical title contrast and return z-ranked results."""
    work = prepare_period_contrast(
        df, focal_start, focal_end, baseline_start, baseline_end,
    )
    work["final_historical_title_features"] = work["title"].map(
        final_historical_title_features,
    )
    return distinctive_features(
        work,
        "final_historical_title_features",
        min_doc_freq=min_document_frequency,
        alpha=correction,
    )


def run_final_historical_title_contrasts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run both and only the two historical contrasts specified in the final SI."""
    return {
        label: final_historical_title_contrast(df, fs, fe, bs, be)
        for label, fs, fe, bs, be in FINAL_HISTORICAL_CONTRASTS
    }


def _normalize_text(text: object) -> str:
    s=str(text or "").lower(); s=re.sub(r"[^a-z0-9\s\-]"," ",s); return re.sub(r"\s+"," ",s).strip()


def tokenize(text: object, stopwords: set[str] | None = None) -> list[str]:
    sw=DEFAULT_STOPWORDS if stopwords is None else stopwords
    return [x for x in _normalize_text(text).split() if len(x)>2 and x not in sw and not x.isdigit()]


def ngrams(tokens: Sequence[str], sizes: Sequence[int]=(1,2)) -> list[str]:
    out=[]
    for n in sizes:
        if n < 1: continue
        out.extend(" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return out


def _keyword_strings(value: object) -> list[str]:
    vals=parse_list_cell(value)
    if len(vals)==1 and (';' in vals[0] or '|' in vals[0]):
        vals=[x.strip() for x in re.split(r";|\|",vals[0]) if x.strip()]
    return vals


def text_feature_set(row: pd.Series, mode: str='title', ngram_sizes: Sequence[int]=(1,2), stopwords: set[str] | None=None) -> set[str]:
    parts=[]
    if mode in {'title','title_abstract_keywords'}: parts.append(row.get('title',''))
    if mode in {'abstract','title_abstract_keywords'}: parts.append(row.get('abstract',''))
    toks=[]
    for p in parts: toks.extend(tokenize(p,stopwords))
    feats=ngrams(toks,ngram_sizes)
    if mode in {'keywords','title_abstract_keywords'}:
        for kw in _keyword_strings(row.get('keywords',[])):
            kt=tokenize(kw,stopwords); feats.extend(ngrams(kt,ngram_sizes));
    return set(feats)


def prepare_period_contrast(
    df: pd.DataFrame,
    focal_start: int,
    focal_end: int,
    baseline_start: int | None,
    baseline_end: int,
) -> pd.DataFrame:
    out=df.copy(); out['year']=pd.to_numeric(out['year'],errors='coerce');out=out[out.year.notna()].copy();out['year']=out.year.astype(int)
    if baseline_start is None: baseline_start=int(out.year.min())
    if focal_start>focal_end or baseline_start>baseline_end: raise ValueError('Invalid period bounds')
    if not (baseline_end < focal_start or focal_end < baseline_start):
        raise ValueError('Baseline and focal periods must not overlap')
    out['period']=np.select([
        out.year.between(int(focal_start),int(focal_end)),
        out.year.between(int(baseline_start),int(baseline_end)),
    ],['focal','baseline'],default='other')
    return out[out.period.isin(['focal','baseline'])].copy()


def add_feature_column(
    df: pd.DataFrame,
    feature: str,
    ngram_sizes: Sequence[int]=(1,2),
    stopwords: set[str] | None=None,
) -> tuple[pd.DataFrame,str]:
    out=df.copy(); col=f'features_{feature}'
    if feature in {'title','abstract','keywords','title_abstract_keywords'}:
        out[col]=out.apply(lambda r:text_feature_set(r,feature,ngram_sizes,stopwords),axis=1)
    elif feature == 'authors':
        src='authors_network' if 'authors_network' in out.columns else 'authors_normalized' if 'authors_normalized' in out.columns else 'authors'
        out[col]=out[src].apply(lambda x:set(parse_list_cell(x)))
    elif feature == 'institutions':
        src='institutions_normalized' if 'institutions_normalized' in out.columns else 'institutions'
        out[col]=out[src].apply(lambda x:set(parse_list_cell(x)))
    elif feature == 'countries':
        src='countries_normalized' if 'countries_normalized' in out.columns else 'countries'
        out[col]=out[src].apply(lambda x:set(parse_list_cell(x)))
    else:
        raise ValueError(f'Unknown feature family: {feature}')
    return out,col


def availability_by_period(df:pd.DataFrame, columns:Sequence[str]) -> pd.DataFrame:
    rows=[]
    for period,g in df.groupby('period'):
        for col in columns:
            if col not in g.columns: continue
            def nonempty(v):
                if isinstance(v,(list,tuple,set,dict)): return len(v)>0
                try:
                    if pd.isna(v): return False
                except Exception: pass
                return bool(str(v).strip())
            rows.append({'period':period,'column':col,'n_records':len(g),'share_nonempty':float(g[col].apply(nonempty).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def _doc_freq_by_group(df:pd.DataFrame, feature_col:str):
    counts=defaultdict(Counter); totals=Counter()
    for _,r in df.iterrows():
        p=r.period; totals[p]+=1
        for feat in r[feature_col]: counts[p][feat]+=1
    return counts,totals


def _log_odds_z(a_count,a_total,b_count,b_total,alpha=.5):
    a=a_count+alpha;b=a_total-a_count+alpha;c=b_count+alpha;d=b_total-b_count+alpha
    lo=math.log(a/b)-math.log(c/d);se=math.sqrt(1/a+1/b+1/c+1/d);return lo,lo/se


def distinctive_features(df:pd.DataFrame, feature_col:str, min_doc_freq:int=3, alpha:float=.5) -> pd.DataFrame:
    counts,totals=_doc_freq_by_group(df,feature_col); rows=[]
    for feat in set(counts['focal'])|set(counts['baseline']):
        fc=counts['focal'][feat];bc=counts['baseline'][feat]
        if fc+bc < min_doc_freq: continue
        lo,z=_log_odds_z(fc,totals['focal'],bc,totals['baseline'],alpha)
        rows.append({'feature':feat,'focal_count':fc,'baseline_count':bc,'focal_share':fc/totals['focal'] if totals['focal'] else np.nan,
                     'baseline_share':bc/totals['baseline'] if totals['baseline'] else np.nan,'log_odds':lo,'z_score':z,
                     'focal_documents':totals['focal'],'baseline_documents':totals['baseline']})
    return pd.DataFrame(rows).sort_values(['z_score','focal_count'],ascending=[False,False]).reset_index(drop=True) if rows else pd.DataFrame()


def yearly_presence_trends(df:pd.DataFrame, feature_col:str, features:Sequence[str], focal_start:int, focal_end:int) -> pd.DataFrame:
    focal=df[df.year.between(focal_start,focal_end)]
    rows=[]
    for feat in features:
        yr=[]
        for y,g in focal.groupby('year'):
            yr.append((int(y),float(g[feature_col].apply(lambda s:feat in s).mean())))
        if len(yr)>=2:
            x=np.array([a for a,_ in yr],dtype=float);v=np.array([b for _,b in yr],dtype=float);res=linregress(x,v)
            rows.append({'feature':feat,'slope':res.slope,'intercept':res.intercept,'r_value':res.rvalue,'r_squared':res.rvalue**2,'p_value':res.pvalue,'std_err':res.stderr,'n_years':len(yr)})
    return pd.DataFrame(rows).sort_values('slope',ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def institution_term_cooccurrence(
    df:pd.DataFrame,
    institution_col:str,
    term_col:str,
    institutions:Sequence[str],
    terms:Sequence[str],
    focal_start:int,
    focal_end:int,
) -> pd.DataFrame:
    focal=df[df.year.between(focal_start,focal_end)];rows=[]
    for inst in institutions:
        for term in terms:
            n=int(focal.apply(lambda r:inst in r[institution_col] and term in r[term_col],axis=1).sum())
            if n: rows.append({'institution':inst,'term':term,'cooccurring_documents':n})
    return pd.DataFrame(rows).sort_values('cooccurring_documents',ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
