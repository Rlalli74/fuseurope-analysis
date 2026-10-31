"""Country-level collaboration and internationalization analyses.

Two country representations are kept distinct:
- ``countries_normalized`` is geographic and is used for the primary ICR because
  ICR is defined as cross-*national* coauthorship.
- ``countries_network`` replaces the UK assignment of a JET affiliation with a
  distinct ``JET`` node and is used for co-country visualizations and for treating
  JET as a special Euratom organizational partner.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence
import ast, json, math
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint


def parse_list_cell(value):
    if isinstance(value,list): return [str(x).strip() for x in value if str(x).strip()]
    if value is None: return []
    try:
        if pd.isna(value): return []
    except Exception: pass
    s=str(value).strip()
    if not s:return []
    for parser in (json.loads,ast.literal_eval):
        try:
            x=parser(s)
            if isinstance(x,(list,tuple)): return [str(v).strip() for v in x if str(v).strip()]
        except Exception: pass
    return [v.strip() for v in s.split(';') if v.strip()]


def unique(seq): return list(dict.fromkeys(x for x in seq if x))

# Entry year into the Community/Euratom political space used in the historical paper.
# Germany is represented as GERMANY in the canonical country normalization.
EURATOM_ENTRY={
    'BELGIUM':1958,'FRANCE':1958,'GERMANY':1958,'ITALY':1958,'LUXEMBOURG':1958,'NETHERLANDS':1958,
    'DENMARK':1973,'IRELAND':1973,'UK':1973,
    'GREECE':1981,'PORTUGAL':1986,'SPAIN':1986,
    'AUSTRIA':1995,'FINLAND':1995,'SWEDEN':1995,
    'CYPRUS':2004,'CZECH REPUBLIC':2004,'ESTONIA':2004,'HUNGARY':2004,'LATVIA':2004,
    'LITHUANIA':2004,'MALTA':2004,'POLAND':2004,'SLOVAKIA':2004,'SLOVENIA':2004,
    'BULGARIA':2007,'ROMANIA':2007,
}

# Fixed country sets used only as compositional robustness checks.  EC12_1986 is
# the principal fixed-set comparison because Figure 10 begins in 1988, after all
# twelve countries had entered the Community.  Applying it to earlier Figure-9
# years is explicitly counterfactual and serves only to hold composition constant.
FIXED_EUROPE_SETS={
    'EC6_1958': frozenset({'BELGIUM','FRANCE','GERMANY','ITALY','LUXEMBOURG','NETHERLANDS'}),
    'EC12_1986': frozenset({'BELGIUM','FRANCE','GERMANY','ITALY','LUXEMBOURG','NETHERLANDS',
                            'DENMARK','IRELAND','UK','GREECE','PORTUGAL','SPAIN'}),
    'EC10_1981': frozenset({'BELGIUM','FRANCE','GERMANY','ITALY','LUXEMBOURG','NETHERLANDS',
                            'DENMARK','IRELAND','UK','GREECE'}),
    'EU15_1995': frozenset({'BELGIUM','FRANCE','GERMANY','ITALY','LUXEMBOURG','NETHERLANDS',
                            'DENMARK','IRELAND','UK','GREECE','PORTUGAL','SPAIN',
                            'AUSTRIA','FINLAND','SWEDEN'}),
    'EU25_2004': frozenset({'BELGIUM','FRANCE','GERMANY','ITALY','LUXEMBOURG','NETHERLANDS',
                            'DENMARK','IRELAND','UK','GREECE','PORTUGAL','SPAIN',
                            'AUSTRIA','FINLAND','SWEDEN','CYPRUS','CZECH REPUBLIC','ESTONIA',
                            'HUNGARY','LATVIA','LITHUANIA','MALTA','POLAND','SLOVAKIA','SLOVENIA'}),
}


def is_euratom_actor(actor:str,year:int)->bool:
    if actor=='JET': return year>=1978  # publication-network convention; JET as multinational Euratom node
    return actor in EURATOM_ENTRY and year>=EURATOM_ENTRY[actor]


def is_european_actor(
    actor: str,
    year: int,
    definition: str = 'dynamic',
    fixed_set: str = 'EC12_1986',
    include_jet: bool = True,
) -> bool:
    """Classify a partner under the historical or a fixed Europe definition."""
    actor=str(actor).upper()
    if actor=='JET':
        return bool(include_jet and year>=1978)
    if definition=='dynamic':
        return actor in EURATOM_ENTRY and year>=EURATOM_ENTRY[actor]
    if definition=='fixed':
        if fixed_set not in FIXED_EUROPE_SETS:
            raise ValueError(f'Unknown fixed Europe set: {fixed_set}')
        return actor in FIXED_EUROPE_SETS[fixed_set]
    raise ValueError("definition must be 'dynamic' or 'fixed'")


def prepare_country_dataframe(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); out['year']=pd.to_numeric(out.year,errors='coerce');out=out[out.year.notna()].copy();out['year']=out.year.astype(int)
    for c in ['countries_normalized','countries_network','authors_network','institutions','document_types']:
        if c in out.columns: out[c]=out[c].apply(parse_list_cell)
    out['geo_countries']=out['countries_normalized'].apply(unique)
    out['network_countries']=out['countries_network'].apply(unique)
    out['has_geo_affiliation']=out.geo_countries.apply(bool)
    out['is_international_geo']=out.geo_countries.apply(lambda x:len(x)>1)
    out['is_international_network']=out.network_countries.apply(lambda x:len(x)>1)
    out['n_geo_countries']=out.geo_countries.apply(len)
    out['n_network_countries']=out.network_countries.apply(len)
    if 'n_affiliations' not in out.columns:
        out['n_affiliations']=out['affiliations'].apply(lambda x:len(parse_list_cell(x))) if 'affiliations' in out else np.nan
    if 'n_authors_network' not in out.columns:
        out['n_authors_network']=out['authors_network'].apply(len) if 'authors_network' in out else np.nan
    return out


def yearly_icr(df:pd.DataFrame,country_col='geo_countries')->pd.DataFrame:
    """Annual International Collaboration Rate with Wilson intervals."""
    rows=[]
    for year,g in df.groupby('year'):
        affiliated=g[g[country_col].apply(bool)]
        intl=affiliated[country_col].apply(lambda x:len(unique(x))>1)
        n=len(affiliated); s=int(intl.sum())
        lo,hi=(np.nan,np.nan) if n==0 else proportion_confint(s,n,alpha=.05,method='wilson')
        rows.append({'year':int(year),'total_papers':len(g),'papers_with_country':n,'papers_without_country':len(g)-n,
                     'coverage_rate':n/len(g) if len(g) else np.nan,'international_papers':s,'ICR':s/n if n else np.nan,
                     'ICR_low':lo,'ICR_high':hi})
    return pd.DataFrame(rows).sort_values('year')


def centered_pooled_ratio(
    annual: pd.DataFrame,
    numerator: str,
    denominator: str,
    *,
    window: int = 3,
    output_name: str = "pooled_ratio",
) -> pd.DataFrame:
    """Pool integer components in complete centered windows before division."""
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("centered pooled windows must be positive and odd")
    d = annual.sort_values("year").copy()
    expected_years = pd.RangeIndex(int(d.year.min()), int(d.year.max()) + 1)
    if list(d.year.astype(int)) != list(expected_years):
        raise ValueError("annual input must contain one row for every consecutive year")
    num = pd.to_numeric(d[numerator], errors="raise").rolling(window, center=True, min_periods=window).sum()
    den = pd.to_numeric(d[denominator], errors="raise").rolling(window, center=True, min_periods=window).sum()
    out = pd.DataFrame({
        "year": d.year.astype(int),
        f"{numerator}_pooled_{window}y": num,
        f"{denominator}_pooled_{window}y": den,
        output_name: np.where(den > 0, num / den, np.nan),
    })
    return out[out[output_name].notna()].reset_index(drop=True)


def icr_with_pooled_smoothing(
    df: pd.DataFrame,
    start: int = 1973,
    end: int = 2007,
    windows: Sequence[int] = (3, 5),
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Return annual Wilson ICR and complete centered pooled-ratio series."""
    annual = yearly_icr(df)
    annual = annual[annual.year.between(int(start), int(end))].reset_index(drop=True)
    pooled = {
        int(window): centered_pooled_ratio(
            annual, "international_papers", "papers_with_country",
            window=int(window), output_name="ICR_pooled",
        )
        for window in windows
    }
    return annual, pooled


def icr_jet_sensitivity(df:pd.DataFrame)->pd.DataFrame:
    a=yearly_icr(df,'geo_countries').rename(columns={'ICR':'ICR_geographic'})[['year','ICR_geographic']]
    b=yearly_icr(df,'network_countries').rename(columns={'ICR':'ICR_JET_node'})[['year','ICR_JET_node']]
    out=a.merge(b,on='year',how='outer');out['difference']=out.ICR_JET_node-out.ICR_geographic
    return out




def _affiliation_bin(x):
    try:x=int(x)
    except:return None
    if x<=1:return '1'
    if x==2:return '2'
    if x==3:return '3'
    if x==4:return '4'
    return '5+'

def _author_bin(x):
    try:x=int(x)
    except:return None
    if x<=1:return '1'
    if x==2:return '2'
    if x<=4:return '3-4'
    if x<=8:return '5-8'
    return '9+'

def standardized_icr(df:pd.DataFrame,start=1973,end=2007)->pd.DataFrame:
    """Direct-standardization diagnostics for changing record/team composition."""
    work=df[(df.year>=start)&(df.year<=end)&df.has_geo_affiliation].copy()
    work['affiliation_bin']=work.n_affiliations.apply(_affiliation_bin)
    work['author_bin']=work.n_authors_network.apply(_author_bin)
    rows=pd.DataFrame({'year':range(start,end+1)})
    for bincol,label in [('affiliation_bin','affiliation_standardized'),('author_bin','author_standardized')]:
        w=work[work[bincol].notna()].copy(); pooled=w[bincol].value_counts(normalize=True)
        rates=w.groupby(['year',bincol]).is_international_geo.mean()
        vals=[]
        for y in range(start,end+1):
            num=den=0.0
            for b,pw in pooled.items():
                if (y,b) in rates.index:
                    num+=pw*float(rates.loc[(y,b)]);den+=pw
            vals.append(num/den if den else np.nan)
        rows[label]=vals
    obs=yearly_icr(df); rows=rows.merge(obs[['year','ICR']],on='year',how='left')
    return rows

def missing_country_bounds(icr_df:pd.DataFrame)->pd.DataFrame:
    out=icr_df.copy()
    out['intl_lower_bound_total']=out.international_papers/out.total_papers
    out['intl_upper_bound_total']=(out.international_papers+out.papers_without_country)/out.total_papers
    return out

def adjusted_internationalization_model(df:pd.DataFrame,start=1973,end=2007):
    work=df[(df.year>=start)&(df.year<=end)&df.has_geo_affiliation].copy()
    work['international']=work.is_international_geo.astype(int)
    work['log_affiliations']=np.log1p(pd.to_numeric(work.n_affiliations,errors='coerce').fillna(0))
    work['log_authors']=np.log1p(pd.to_numeric(work.n_authors_network,errors='coerce').fillna(0))
    # A spline year term avoids imposing linearity in this optional adjusted diagnostic.
    model=smf.glm('international ~ bs(year, df=6, degree=3) + log_affiliations + log_authors',data=work,family=sm.families.Binomial()).fit(cov_type='HC1')
    grid=pd.DataFrame({'year':range(start,end+1),'log_affiliations':work.log_affiliations.median(),'log_authors':work.log_authors.median()})
    grid['adjusted_probability']=model.predict(grid)
    return model,grid


def build_cocountry_graph(df,start,end,use_network_countries=True,paper_weight_col=None):
    col='network_countries' if use_network_countries else 'geo_countries'
    sub=df[(df.year>=start)&(df.year<=end)]
    G=nx.Graph()
    for _,r in sub.iterrows():
        cs=unique(r[col]); n=len(cs)
        paper_weight=1.0
        if paper_weight_col is not None:
            try: paper_weight=float(r.get(paper_weight_col,1.0))
            except Exception: paper_weight=1.0
            if not np.isfinite(paper_weight) or paper_weight<=0: paper_weight=1.0
        for c in cs:G.add_node(c)
        if n<2:continue
        frac=paper_weight/(n-1)
        for a,b in combinations(cs,2):
            if not G.has_edge(a,b):G.add_edge(a,b,weight=0,paper_count=0,fractional_weight=0.0)
            G[a][b]['weight']+=paper_weight;G[a][b]['paper_count']+=1;G[a][b]['fractional_weight']+=frac
    return G




def largest_component(G: nx.Graph)->nx.Graph:
    if G.number_of_nodes()==0:return nx.Graph()
    nodes=max(nx.connected_components(G),key=len)
    return G.subgraph(nodes).copy()

def cocountry_node_table(G,year_for_membership=None):
    if len(G)==0:return pd.DataFrame()
    bc=nx.betweenness_centrality(G,normalized=True,weight=None)
    rows=[]
    for n in G:
        rows.append({'actor':n,'degree':G.degree(n),'weighted_degree':G.degree(n,weight='weight'),'fractional_degree':G.degree(n,weight='fractional_weight'),
                     'betweenness':bc[n],'euratom_actor':is_euratom_actor(n,year_for_membership) if year_for_membership else np.nan})
    return pd.DataFrame(rows).sort_values('betweenness',ascending=False)


def cocountry_brokerage(
    df: pd.DataFrame,
    start: int,
    end: int,
    *,
    modified_jet_representation: bool = True,
) -> tuple[nx.Graph, nx.Graph, pd.DataFrame]:
    """Build a co-country graph and compute unweighted betweenness on its LCC."""
    graph = build_cocountry_graph(df, start, end, modified_jet_representation)
    lcc = largest_component(graph)
    table = cocountry_node_table(lcc, end)
    table["start_year"] = int(start)
    table["end_year"] = int(end)
    table["representation"] = "modified_JET_node" if modified_jet_representation else "strictly_geographic"
    table["betweenness_topology"] = "unweighted_LCC"
    return graph, lcc, table


def cocountry_jet_geographic_benchmark(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Return actor-level modified-versus-geographic brokerage comparison."""
    _, _, modified = cocountry_brokerage(df, start, end, modified_jet_representation=True)
    _, _, geographic = cocountry_brokerage(df, start, end, modified_jet_representation=False)
    left = modified[["actor", "betweenness"]].rename(columns={"betweenness": "modified_betweenness"})
    right = geographic[["actor", "betweenness"]].rename(columns={"betweenness": "geographic_betweenness"})
    out = left.merge(right, on="actor", how="outer").fillna(0.0)
    out["start_year"], out["end_year"] = int(start), int(end)
    return out.sort_values(["modified_betweenness", "actor"], ascending=[False, True]).reset_index(drop=True)


def european_member_set(year: int, definition: str = "dynamic", fixed_set: str | None = None) -> frozenset[str]:
    if definition == "dynamic":
        return frozenset(country for country, entry in EURATOM_ENTRY.items() if int(year) >= entry)
    if definition == "fixed" and fixed_set in FIXED_EUROPE_SETS:
        return FIXED_EUROPE_SETS[str(fixed_set)]
    raise ValueError("Use dynamic membership or a named fixed European set")


def european_indicator_components(
    df: pd.DataFrame,
    *,
    start: int = 1973,
    end: int = 2007,
    membership_definition: str = "dynamic",
    fixed_set: str | None = None,
) -> pd.DataFrame:
    """Calculate annual components for the three final European indicators."""
    rows = []
    work = df[df.year.between(int(start), int(end))]
    for year, all_papers in work.groupby("year"):
        members = european_member_set(int(year), membership_definition, fixed_set)
        coded = all_papers[all_papers.geo_countries.apply(bool)]
        output_num = 0.0
        partner_num = 0.0
        partner_den = 0.0
        edge_weights: dict[tuple[str, str], float] = {}
        for countries in coded.geo_countries:
            countries = sorted(unique(countries))
            k = len(countries)
            european_count = len(set(countries) & set(members))
            output_num += european_count / k
            if k < 2:
                continue
            partner_den += european_count
            if european_count >= 2:
                partner_num += european_count
            pairs = list(combinations(countries, 2))
            pair_weight = 1.0 / len(pairs)
            for pair in pairs:
                edge_weights[pair] = edge_weights.get(pair, 0.0) + pair_weight
        strengths: dict[str, float] = {}
        observed = 0.0
        total_weight = float(sum(edge_weights.values()))
        for (a, b), weight in edge_weights.items():
            strengths[a] = strengths.get(a, 0.0) + weight
            strengths[b] = strengths.get(b, 0.0) + weight
            if a in members and b in members:
                observed += weight
        european_strengths = [strengths.get(country, 0.0) for country in members]
        s_e = float(sum(european_strengths))
        q_e = float(sum(value * value for value in european_strengths))
        expected = ((s_e * s_e - q_e) / (4.0 * total_weight)) if total_weight > 0 else 0.0
        rows.append({
            "year": int(year),
            "analytical_publications": int(len(all_papers)),
            "country_coded": int(len(coded)),
            "european_fractional_output_numerator": output_num,
            "european_fractional_output_denominator": int(len(coded)),
            "european_partner_numerator": partner_num,
            "european_partner_denominator": partner_den,
            "european_observed_dyadic_weight": observed,
            "european_expected_dyadic_weight": expected,
            "international_dyadic_weight": total_weight,
            "membership_definition": membership_definition,
            "fixed_set": fixed_set or "",
        })
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return _add_european_ratios(out)


def _add_european_ratios(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["coverage"] = np.where(out.analytical_publications > 0, out.country_coded / out.analytical_publications, np.nan)
    out["european_fractional_output_share"] = np.where(
        out.european_fractional_output_denominator > 0,
        out.european_fractional_output_numerator / out.european_fractional_output_denominator,
        np.nan,
    )
    out["conditional_european_partner_share"] = np.where(
        out.european_partner_denominator > 0,
        out.european_partner_numerator / out.european_partner_denominator,
        np.nan,
    )
    out["oe_european_affinity"] = np.where(
        out.european_expected_dyadic_weight > 0,
        out.european_observed_dyadic_weight / out.european_expected_dyadic_weight,
        np.nan,
    )
    return out


EUROPEAN_COMPONENT_COLUMNS = (
    "analytical_publications", "country_coded",
    "european_fractional_output_numerator", "european_fractional_output_denominator",
    "european_partner_numerator", "european_partner_denominator",
    "european_observed_dyadic_weight", "european_expected_dyadic_weight",
    "international_dyadic_weight",
)


def pool_european_indicators(
    annual: pd.DataFrame,
    periods: Sequence[tuple[str, int, int]],
) -> pd.DataFrame:
    """Pool annual components, preserving year-specific dynamic membership."""
    rows = []
    for label, start, end in periods:
        selected = annual[annual.year.between(int(start), int(end))]
        row = {"period": label, "start_year": int(start), "end_year": int(end)}
        row.update({column: float(selected[column].sum()) for column in EUROPEAN_COMPONENT_COLUMNS})
        rows.append(row)
    return _add_european_ratios(pd.DataFrame(rows))


def smooth_european_indicators(annual: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Centered complete-window pooling for all three European measures."""
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("centered pooled windows must be positive and odd")
    ordered = annual.sort_values("year").reset_index(drop=True)
    pooled = ordered[list(EUROPEAN_COMPONENT_COLUMNS)].rolling(
        window, center=True, min_periods=window
    ).sum()
    pooled.insert(0, "year", ordered.year)
    pooled = pooled.dropna(subset=["analytical_publications"]).reset_index(drop=True)
    return _add_european_ratios(pooled)


def enlargement_definition_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    comparisons = {
        1986: ("EC12_1986", "EC10_1981"),
        1995: ("EU15_1995", "EC12_1986"),
        2004: ("EU25_2004", "EU15_1995"),
    }
    rows = []
    for year, (dynamic_set, legacy_set) in comparisons.items():
        for label, fixed_set in (("dynamic_set", dynamic_set), ("legacy_core", legacy_set)):
            result = european_indicator_components(
                df, start=year, end=year, membership_definition="fixed", fixed_set=fixed_set
            ).iloc[0]
            rows.append({
                "year": year, "definition": label, "membership_set": fixed_set,
                "european_fractional_output_share": result.european_fractional_output_share,
                "conditional_european_partner_share": result.conditional_european_partner_share,
                "oe_european_affinity": result.oe_european_affinity,
            })
    return pd.DataFrame(rows)


def focal_country_intra_extra(
    df,country,start=1973,end=2007,
    europe_definition='dynamic',fixed_europe_set='EC12_1986',include_jet_partner=True,
):
    """Paper-level intra-/extra-Community indicators for a focal national actor.

    JET can count as an Euratom partner through ``network_countries``. The focal
    country itself must be present as a national node; JET-only papers do not become
    UK papers merely because JET is physically at Culham.
    """
    rows=[]
    sub=df[(df.year>=start)&(df.year<=end)]
    for _,r in sub.iterrows():
        y=int(r.year);actors=unique(r.network_countries)
        if country not in actors:continue
        partners=[a for a in actors if a!=country]
        if not include_jet_partner:
            partners=[a for a in partners if a!='JET']
        eur=[a for a in partners if is_european_actor(a,y,europe_definition,fixed_europe_set,include_jet_partner)]
        ext=[a for a in partners if not is_european_actor(a,y,europe_definition,fixed_europe_set,include_jet_partner)]
        rows.append({'year':y,'country':country,'is_international':bool(partners),'has_euratom_partner':bool(eur),'has_extra_partner':bool(ext),
                     'intra_euratom_only':bool(eur) and not bool(ext),'mixed_euratom_extra':bool(eur) and bool(ext)})
    pap=pd.DataFrame(rows)
    if pap.empty:return pap,pd.DataFrame()
    agg=pap.groupby('year').agg(total_papers=('country','size'),international_papers=('is_international','sum'),
        papers_with_euratom_partner=('has_euratom_partner','sum'),papers_with_extra_partner=('has_extra_partner','sum'),
        intra_euratom_only=('intra_euratom_only','sum'),mixed_euratom_extra=('mixed_euratom_extra','sum')).reset_index()
    for c in ['international_papers','papers_with_euratom_partner','papers_with_extra_partner','intra_euratom_only','mixed_euratom_extra']:
        agg[c+'_share']=agg[c]/agg.total_papers
    return pap,agg


def country_metadata_bias_audit(df:pd.DataFrame,start=1973,end=2007,period_width=5):
    """Audit C1 coverage by corresponding-author country recorded in RP.

    RP identifies only the corresponding author's address.  It is therefore used
    to diagnose differential C1 missingness, never to infer that a paper is
    domestic or to reconstruct missing co-country edges.
    """
    work=df[(df.year>=start)&(df.year<=end)].copy()
    if 'reprint_country_normalized' not in work.columns:
        raise ValueError('reprint_country_normalized is absent; rebuild phases 01-03 with the RP-preserving parser')
    work['rp_country']=work.reprint_country_normalized.fillna('').astype(str).str.upper().str.strip()
    work['rp_group']=np.where(work.rp_country.eq('USA'),'USA',np.where(work.rp_country.ne(''),'NON_USA','UNRESOLVED'))
    work['has_c1_country']=work.geo_countries.apply(bool)
    work['period_start']=start+((work.year-start)//int(period_width))*int(period_width)
    work['period_end']=np.minimum(work.period_start+int(period_width)-1,end)
    def summarize(keys):
        out=(work.groupby(keys,dropna=False).agg(
            papers=('year','size'),papers_with_c1=('has_c1_country','sum')
        ).reset_index())
        out['c1_coverage']=out.papers_with_c1/out.papers
        cis=[proportion_confint(int(s),int(n),alpha=.05,method='wilson') for s,n in zip(out.papers_with_c1,out.papers)]
        out['coverage_wilson_low']=[x[0] for x in cis];out['coverage_wilson_high']=[x[1] for x in cis]
        return out
    annual=summarize(['year','rp_group'])
    periods=summarize(['period_start','period_end','rp_group'])
    comparable=periods[periods.rp_group.isin(['USA','NON_USA'])].pivot(
        index=['period_start','period_end'],columns='rp_group',values='c1_coverage'
    ).reset_index()
    for col in ['USA','NON_USA']:
        if col not in comparable: comparable[col]=np.nan
    comparable['usa_minus_nonusa_coverage']=comparable['USA']-comparable['NON_USA']
    work['observed_usa']=work.geo_countries.apply(lambda x:'USA' in x)
    work['rp_only_usa']=(~work.has_c1_country)&work.rp_country.eq('USA')
    lower=(work.groupby('year').agg(
        total_papers=('year','size'),observed_usa_papers=('observed_usa','sum'),
        c1_missing_rp_usa=('rp_only_usa','sum'),papers_with_c1=('has_c1_country','sum')
    ).reset_index())
    lower['observed_usa_share_all']=lower.observed_usa_papers/lower.total_papers
    lower['minimum_usa_share_with_rp_recovery']=(lower.observed_usa_papers+lower.c1_missing_rp_usa)/lower.total_papers
    return annual,periods,comparable,lower


def c1_inverse_probability_weights(df:pd.DataFrame,start=1973,end=2007,period_width=5,min_probability=.10,max_weight=10.0):
    """Return diagnostic paper weights for differential C1 coverage by RP country.

    The weights correct representation by corresponding-author country and period
    among records with an RP country.  They cannot recover unidentified partner
    countries and therefore support only a sensitivity analysis, not replacement
    of the complete-case network.
    """
    work=df[(df.year>=start)&(df.year<=end)].copy()
    if 'reprint_country_normalized' not in work.columns:
        raise ValueError('reprint_country_normalized is absent; rebuild phases 01-03 first')
    work['rp_country']=work.reprint_country_normalized.fillna('').astype(str).str.upper().str.strip()
    work['rp_group']=np.where(work.rp_country.eq('USA'),'USA',np.where(work.rp_country.ne(''),'NON_USA','UNRESOLVED'))
    work['period_start']=start+((work.year-start)//int(period_width))*int(period_width)
    work['has_c1_country']=work.geo_countries.apply(bool)
    rates=work.groupby(['period_start','rp_group']).has_c1_country.mean().rename('c1_probability').reset_index()
    work=work.merge(rates,on=['period_start','rp_group'],how='left')
    work['c1_probability']=work.c1_probability.clip(lower=float(min_probability),upper=1.0)
    work['c1_ipw']=np.where(work.has_c1_country,(1.0/work.c1_probability).clip(upper=float(max_weight)),np.nan)
    cols=[c for c in ['record_id','ut','year','rp_country','rp_group','has_c1_country','c1_probability','c1_ipw'] if c in work.columns]
    return work[cols]


def us_address_heuristic_audit(affiliation_df:pd.DataFrame):
    """Quantify how many U.S. assignments rely on state/ZIP heuristics.

    This check separates a possible parser asymmetry from the much larger problem
    of records that lack C1 entirely.  The affiliation table produced by phase 02
    contains the required normalization-method provenance.
    """
    required={'record_id','country_normalized','country_normalization_method'}
    missing=required-set(affiliation_df.columns)
    if missing: raise ValueError(f'affiliation table lacks columns: {sorted(missing)}')
    x=affiliation_df.copy()
    x['is_usa']=x.country_normalized.fillna('').astype(str).str.upper().eq('USA')
    methods={'us_state_zip','us_state_only','us_puerto_rico'}
    x['usa_state_heuristic']=x.country_normalization_method.isin(methods)
    summary=pd.DataFrame([{
        'affiliation_lines':len(x),
        'usa_lines':int(x.is_usa.sum()),
        'usa_lines_using_state_heuristic':int(x.usa_state_heuristic.sum()),
        'share_of_usa_lines_using_state_heuristic':float(x.usa_state_heuristic.sum()/x.is_usa.sum()) if x.is_usa.sum() else np.nan,
        'records_with_usa':int(x.loc[x.is_usa,'record_id'].nunique()),
        'records_with_only_heuristic_usa_evidence':int(x.loc[x.usa_state_heuristic,'record_id'].nunique()),
    }])
    by_year=(x.groupby('year',dropna=False).agg(
        affiliation_lines=('record_id','size'),usa_lines=('is_usa','sum'),
        usa_lines_using_state_heuristic=('usa_state_heuristic','sum')
    ).reset_index())
    by_year['share_of_usa_lines_using_state_heuristic']=np.where(
        by_year.usa_lines>0,by_year.usa_lines_using_state_heuristic/by_year.usa_lines,np.nan
    )
    return summary,by_year


def focal_country_logistic_trend(paper_df):
    if paper_df.empty:return None
    x=paper_df.copy();x['t']=x.year-x.year.min();x['y']=x.has_euratom_partner.astype(int)
    if x.y.nunique()<2:return None
    return smf.glm('y ~ t',data=x,family=sm.families.Binomial()).fit(cov_type='HC1')


def country_fractional_output(df,country_col='geo_countries'):
    rows=[]
    for _,r in df.iterrows():
        cs=unique(r[country_col]);
        if not cs:continue
        w=1/len(cs)
        for c in cs:rows.append({'year':int(r.year),'country':c,'fractional_papers':w,'full_count_papers':1})
    if not rows:return pd.DataFrame()
    return pd.DataFrame(rows).groupby(['year','country'],as_index=False).sum()


def country_time_windows(start_year:int,end_year:int,mode:str='annual',window_size:int=1,step:int=1):
    """Country-network equivalent of the generic author-network window helper."""
    from .exploration import make_time_windows
    return make_time_windows(start_year,end_year,mode,window_size,step)


def cocountry_longitudinal_metrics(
    df:pd.DataFrame,
    start_year:int,
    end_year:int,
    mode:str='annual',
    window_size:int=1,
    step:int=1,
    metrics:Sequence[str]=('number_of_nodes','number_of_edges','average_degree','density','largest_component_size','ratio_largest_component_nodes'),
    use_network_countries:bool=True,
    modularity_method:str='louvain',
    seed:int=42,
)->pd.DataFrame:
    """User-selectable longitudinal metrics for co-country networks."""
    from .exploration import make_time_windows,compute_graph_metrics
    rows=[]
    for w in make_time_windows(start_year,end_year,mode,window_size,step):
        G=build_cocountry_graph(df,w.start_year,w.end_year,use_network_countries)
        row=compute_graph_metrics(G,metrics,modularity_method=modularity_method,seed=seed)
        row.update({'label':w.label,'start_year':w.start_year,'end_year':w.end_year,'window_mode':mode,'window_size':window_size,
                    'country_representation':'countries_network' if use_network_countries else 'countries_normalized'})
        rows.append(row)
    return pd.DataFrame(rows)


def country_centrality_over_time(
    df:pd.DataFrame,
    start_year:int,
    end_year:int,
    measures:Sequence[str]=('betweenness',),
    countries:Sequence[str]|None=None,
    top_n:int|None=None,
    mode:str='annual',
    window_size:int=1,
    step:int=1,
    scope:str='full',
    use_network_countries:bool=True,
    seed:int=42,
    betweenness_exact_max_nodes:int=1000,
    betweenness_samples:int=250,
)->pd.DataFrame:
    """Centralities of selected countries/JET over arbitrary time windows."""
    from .exploration import make_time_windows,compute_centralities
    rows=[]; focal=list(countries or [])
    for w in make_time_windows(start_year,end_year,mode,window_size,step):
        G=build_cocountry_graph(df,w.start_year,w.end_year,use_network_countries)
        cent=compute_centralities(G,measures,scope=scope,seed=seed,betweenness_exact_max_nodes=betweenness_exact_max_nodes,betweenness_samples=betweenness_samples)
        if focal:
            ci=cent.set_index('node') if not cent.empty else pd.DataFrame()
            for actor in focal:
                row={'country':actor,'label':w.label,'start_year':w.start_year,'end_year':w.end_year,'centrality_scope':scope}
                for measure in measures:
                    row[measure]=float(ci.loc[actor,measure]) if (not ci.empty and actor in ci.index) else 0.0
                rows.append(row)
        elif top_n:
            for measure in measures:
                for rank,(_,r) in enumerate(cent.nlargest(int(top_n),measure).iterrows(),1):
                    rows.append({'country':r.node,'measure':measure,'value':float(r[measure]),'rank':rank,'label':w.label,
                                 'start_year':w.start_year,'end_year':w.end_year,'centrality_scope':scope})
        else:
            temp=cent.rename(columns={'node':'country'}).copy();temp['label']=w.label;temp['start_year']=w.start_year;temp['end_year']=w.end_year
            rows.extend(temp.to_dict('records'))
    return pd.DataFrame(rows)


def focal_country_icr(df:pd.DataFrame,country:str,start:int=1973,end:int=2007)->pd.DataFrame:
    """Yearly international-collaboration share among papers containing a focal country."""
    rows=[]
    sub=df[(df.year>=start)&(df.year<=end)]
    for year,g in sub.groupby('year'):
        cg=g[g.geo_countries.apply(lambda x:country in x)]
        if cg.empty: continue
        n=len(cg); s=int(cg.is_international_geo.sum())
        lo,hi=proportion_confint(s,n,alpha=.05,method='wilson') if n else (np.nan,np.nan)
        rows.append({'year':int(year),'country':country,'total_papers':n,'international_papers':s,'ICR':s/n if n else np.nan,'ICR_low':lo,'ICR_high':hi})
    return pd.DataFrame(rows)


def country_year_fractional_stats(df:pd.DataFrame,start:int=1973,end:int=2007,country_col='geo_countries')->pd.DataFrame:
    """Legacy diagnostic decomposition of national output and international propensity."""
    rows=[]
    work=df[(df.year>=start)&(df.year<=end)&df[country_col].apply(bool)]
    for _,r in work.iterrows():
        cs=unique(r[country_col]); k=len(cs)
        if not k: continue
        frac=1.0/k; international=len(cs)>1
        for c in cs:
            rows.append({'year':int(r.year),'country':c,'frac_total':frac,'frac_international':frac if international else 0.0,'frac_domestic':0.0 if international else frac})
    if not rows:return pd.DataFrame(columns=['year','country','frac_total','frac_international','frac_domestic','intl_propensity','year_total_frac','output_share'])
    out=pd.DataFrame(rows).groupby(['year','country'],as_index=False)[['frac_total','frac_international','frac_domestic']].sum()
    out['intl_propensity']=np.where(out.frac_total>0,out.frac_international/out.frac_total,0.0)
    totals=out.groupby('year',as_index=False).frac_total.sum().rename(columns={'frac_total':'year_total_frac'})
    out=out.merge(totals,on='year',how='left');out['output_share']=out.frac_total/out.year_total_frac
    return out


def annual_internationalization_system_metrics(country_year_df:pd.DataFrame)->pd.DataFrame:
    if country_year_df.empty:return pd.DataFrame()
    rows=[]
    for year,g in country_year_df.groupby('year'):
        total_intl=float(g.frac_international.sum()); active_intl=int((g.frac_international>0).sum())
        if total_intl>0:
            shares=g.frac_international/total_intl; hhi=float((shares**2).sum());top5=float(shares.nlargest(5).sum())
        else:hhi=top5=np.nan
        rows.append({'year':int(year),'active_countries':int(g.country.nunique()),'active_international_countries':active_intl,
                     'total_fractional_output':float(g.frac_total.sum()),'total_fractional_international':total_intl,
                     'total_fractional_domestic':float(g.frac_domestic.sum()),'mean_intl_per_active_intl_country':total_intl/active_intl if active_intl else np.nan,
                     'intl_hhi':hhi,'intl_top5_share':top5})
    return pd.DataFrame(rows)


def aggregate_country_stats_by_period(country_year_df:pd.DataFrame,periods:Sequence[tuple[str,int,int]])->pd.DataFrame:
    frames=[]
    for label,start,end in periods:
        g=country_year_df[(country_year_df.year>=start)&(country_year_df.year<=end)]
        if g.empty:continue
        cp=g.groupby('country',as_index=False)[['frac_total','frac_international','frac_domestic']].sum();cp['period']=label
        total=float(cp.frac_total.sum());cp['period_total_frac']=total;cp['output_share']=cp.frac_total/total if total else np.nan;cp['intl_propensity']=np.where(cp.frac_total>0,cp.frac_international/cp.frac_total,0)
        frames.append(cp)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def decompose_country_change(country_period_df:pd.DataFrame,period1:str,period2:str)->pd.DataFrame:
    d1=country_period_df[country_period_df.period==period1].copy();d2=country_period_df[country_period_df.period==period2].copy()
    merged=d1.merge(d2,on='country',how='outer',suffixes=('_p1','_p2')).fillna(0)
    merged['composition_contribution']=(merged.output_share_p2-merged.output_share_p1)*((merged.intl_propensity_p1+merged.intl_propensity_p2)/2)
    merged['propensity_contribution']=(merged.intl_propensity_p2-merged.intl_propensity_p1)*((merged.output_share_p1+merged.output_share_p2)/2)
    merged['total_contribution']=merged.composition_contribution+merged.propensity_contribution
    merged['delta_frac_international']=merged.frac_international_p2-merged.frac_international_p1
    merged['delta_frac_domestic']=merged.frac_domestic_p2-merged.frac_domestic_p1
    merged['delta_intl_propensity']=merged.intl_propensity_p2-merged.intl_propensity_p1
    return merged.sort_values('total_contribution',ascending=False).reset_index(drop=True)


def dyad_year_fractional_stats(df:pd.DataFrame,start:int=1973,end:int=2007,country_col='geo_countries')->pd.DataFrame:
    rows=[]
    for _,r in df[(df.year>=start)&(df.year<=end)].iterrows():
        cs=sorted(unique(r[country_col]))
        if len(cs)<2:continue
        pairs=list(combinations(cs,2));w=1.0/len(pairs)
        for a,b in pairs:rows.append({'year':int(r.year),'country1':a,'country2':b,'pair_weight':w})
    if not rows:return pd.DataFrame(columns=['year','country1','country2','pair_weight'])
    return pd.DataFrame(rows).groupby(['year','country1','country2'],as_index=False).pair_weight.sum()


def aggregate_dyads_by_period(dyad_year_df:pd.DataFrame,periods:Sequence[tuple[str,int,int]])->pd.DataFrame:
    frames=[]
    for label,start,end in periods:
        g=dyad_year_df[(dyad_year_df.year>=start)&(dyad_year_df.year<=end)].groupby(['country1','country2'],as_index=False).pair_weight.sum();g['period']=label;frames.append(g)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def dyad_change(dyad_period_df:pd.DataFrame,period1:str,period2:str)->pd.DataFrame:
    d1=dyad_period_df[dyad_period_df.period==period1];d2=dyad_period_df[dyad_period_df.period==period2]
    m=d1.merge(d2,on=['country1','country2'],how='outer',suffixes=('_p1','_p2')).fillna(0);m['delta_pair_weight']=m.pair_weight_p2-m.pair_weight_p1
    return m.sort_values('delta_pair_weight',ascending=False).reset_index(drop=True)


def build_periods_from_internal_breaks(start_year: int, end_year: int, internal_breaks: Sequence[int]):
    """Convert sorted internal break years into inclusive labeled periods."""
    breaks = sorted({int(x) for x in internal_breaks if start_year <= int(x) < end_year})
    periods=[]; current=int(start_year)
    for br in breaks:
        periods.append((f"P{len(periods)+1}: {current}-{br}", current, br))
        current=br+1
    periods.append((f"P{len(periods)+1}: {current}-{int(end_year)}", current, int(end_year)))
    return periods


def period_summary_icr(df: pd.DataFrame, periods: Sequence[tuple[str,int,int]]) -> pd.DataFrame:
    """Summarize observed ICR and coverage for arbitrary historical periods."""
    rows=[]
    for label,start,end in periods:
        g=df[(df.year>=start)&(df.year<=end)]
        aff=g[g.has_geo_affiliation]
        rows.append({
            "period":label,"start_year":int(start),"end_year":int(end),
            "total_papers":int(len(g)),"affiliated_papers":int(len(aff)),
            "international_papers":int(aff.is_international_geo.sum()) if len(aff) else 0,
            "ICR":float(aff.is_international_geo.mean()) if len(aff) else np.nan,
            "coverage_rate":float(len(aff)/len(g)) if len(g) else np.nan,
            "mean_affiliations":float(pd.to_numeric(g.n_affiliations,errors='coerce').mean()) if len(g) else np.nan,
            "mean_unique_countries_affiliated":float(aff.n_geo_countries.mean()) if len(aff) else np.nan,
        })
    return pd.DataFrame(rows)


# Deprecated compatibility aliases. Current scripts, documentation, and output
# schemas use ICR. These wrappers preserve the earlier API and column names for
# callers outside the final article workflow without changing the calculation.
def _legacy_icr_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={name: name.replace("ICR", "AICI") for name in frame.columns if "ICR" in name})


def yearly_aici(df: pd.DataFrame, country_col="geo_countries") -> pd.DataFrame:
    warnings.warn("yearly_aici is deprecated; use yearly_icr", DeprecationWarning, stacklevel=2)
    return _legacy_icr_columns(yearly_icr(df, country_col))


def aici_with_pooled_smoothing(df: pd.DataFrame, start=1973, end=2007, windows=(3, 5)):
    warnings.warn("aici_with_pooled_smoothing is deprecated; use icr_with_pooled_smoothing", DeprecationWarning, stacklevel=2)
    annual, pooled = icr_with_pooled_smoothing(df, start, end, windows)
    return _legacy_icr_columns(annual), {key: _legacy_icr_columns(value) for key, value in pooled.items()}


def aici_jet_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    warnings.warn("aici_jet_sensitivity is deprecated; use icr_jet_sensitivity", DeprecationWarning, stacklevel=2)
    return _legacy_icr_columns(icr_jet_sensitivity(df))


def standardized_aici(df: pd.DataFrame, start=1973, end=2007) -> pd.DataFrame:
    warnings.warn("standardized_aici is deprecated; use standardized_icr", DeprecationWarning, stacklevel=2)
    return _legacy_icr_columns(standardized_icr(df, start, end))


def focal_country_aici(df: pd.DataFrame, country: str, start=1973, end=2007) -> pd.DataFrame:
    warnings.warn("focal_country_aici is deprecated; use focal_country_icr", DeprecationWarning, stacklevel=2)
    return _legacy_icr_columns(focal_country_icr(df, country, start, end))


def period_summary_aici(df: pd.DataFrame, periods) -> pd.DataFrame:
    warnings.warn("period_summary_aici is deprecated; use period_summary_icr", DeprecationWarning, stacklevel=2)
    return _legacy_icr_columns(period_summary_icr(df, periods))
