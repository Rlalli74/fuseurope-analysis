#!/usr/bin/env python3
"""Assemble aggregate-only final acceptance results from a saved licensed run.

The current 46-column post-disambiguation corpus is the single preserved
retrieval-level representation used by the two paper workflows.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusion_pipeline.corpus import parse_list_cell
from fusion_pipeline.normalization import strip_rp_author_prefix

def readj(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def pct(x): return float(x) * 100.0
def r1(x): return round(float(x), 1)
def r2(x): return round(float(x), 2)
def r3(x): return round(float(x), 3)
def r4(x): return round(float(x), 4)
def rows(df, key): return {str(r[key]): r for _, r in df.iterrows()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    r=Path(args.run_root)
    preserved=r/'03_disambiguation/preserved_identity_corpus_phase3.csv'
    analytical_path=r/'03c_analytical_corpus/analytical_corpus_phase3c.csv'
    p3_path=r/'03_disambiguation/preserved_identity_corpus_phase3.csv'

    p3=pd.read_csv(p3_path, low_memory=False)
    a=pd.read_csv(analytical_path, low_memory=False)
    # Use the same current 46-column file as the publication-trend analysis.
    # No predecessor or external reference CSV is required.
    if len(p3.columns) != 46 or "identity_institutions_normalized" not in p3.columns:
        raise ValueError("Expected the final 46-column post-disambiguation corpus")
    if not p3.ut.is_unique:
        raise ValueError("Duplicate WoS UTs in current preserved corpus")
    p3s=readj(r/'03_disambiguation/phase3_validation_summary.json')
    ids=readj(r/'03b_identity_review_validation/author_identity_review_validation_summary.json')
    cs=readj(r/'03c_analytical_corpus/analytical_corpus_validation_summary.json')

    snap=pd.read_csv(r/'04b_snapshot_networks/selected_period_network_metrics.csv')
    snapmap=rows(snap,'period')
    teams=pd.read_csv(r/'04a_network_diagnostics/yearly_team_size_diagnostics.csv')
    teammap={int(x.year):x for _,x in teams.iterrows()}
    annual=pd.read_csv(r/'04a_network_diagnostics/coauthorship_longitudinal_metrics.csv')
    annualmap={int(x.start_year):x for _,x in annual.iterrows()}

    def snapshot(y):
        x=snapmap[str(y)]; t=teammap[y]
        return {'papers':int(x.papers),'authors':int(x.number_of_nodes),'edges':int(x.number_of_edges),
                'components':int(x.number_of_connected_components),'lcc_authors':int(x.largest_component_size),
                'lcc_share_pct':r1(pct(x.ratio_largest_component_nodes)),'mean_authors_per_paper':r2(t.mean_authors),
                'max_team_size':int(t.max_authors)}

    ts=pd.read_csv(r/'04c_network_robustness/snapshot_team_size_sensitivity.csv')
    ds=pd.read_csv(r/'04c_network_robustness/snapshot_document_type_sensitivity.csv')
    def sensitivity(df, periods, field, order):
        out={'columns':order}
        for y in periods:
            g=df[df.period.astype(str).eq(str(y))]
            lookup={str(x[field]):r1(pct(x.lcc_share_nodes)) for _,x in g.iterrows()}
            out[y]=[lookup[k] for k in order]
        return out
    team_order=['none','100','50','20','10']
    dt_order=['unrestricted_analytical_corpus','exclude_meeting_abstracts','exclude_proceedings_and_meeting_abstracts','articles_only']
    team_sens=sensitivity(ts,[1979,1983,1986,1996,2001],'max_team_size',team_order)
    team_sens['columns']=['all','le100','le50','le20','le10']
    doc_sens=sensitivity(ds,[1979,1983,1986,1996,2001],'document_type_mode',dt_order)
    doc_sens['columns']=['unrestricted','exclude_meeting_abstracts','exclude_proceedings_and_meeting_abstracts','articles_only']

    seed=pd.read_csv(r/'06b_louvain_seed_robustness/louvain_seed_titleword_sensitivity.csv')
    seed_out={}
    for y,g in seed.groupby('year'):
        seed_out[int(y)]={'community_count_range':[int(g.n_communities.min()),int(g.n_communities.max())],
            'min_top20_overlap_vs_seed42_pct':r1(pct(g.top20_overlap_with_reference.min())),
            'max_js':r4(g.js_divergence_from_reference.max())}

    rep=pd.read_csv(r/'06_socio_epistemic/lcc_vs_whole_field_titleword_comparison.csv')
    rep_out={int(x.year):{'top20_overlap_pct':r1(pct(x.top_n_overlap_rate)),'jaccard':r3(x.jaccard),'js':r3(x.js_divergence_full_profiles)} for _,x in rep.iterrows()}
    sim=pd.read_csv(r/'06_socio_epistemic/titleword_snapshot_similarity_weighting_sensitivity.csv')
    def sim_out(profile, full=True):
        d={}
        for _,x in sim[sim.profile_type.eq(profile)].iterrows():
            key=f'{int(x.earlier_year)}-{int(x.later_year)}'
            v={'top20_overlap_pct':r1(pct(x.top_n_overlap_rate)),'js':r3(x.js_divergence_full_profiles)}
            if full: v.update(shared_terms=int(x.shared_terms),jaccard=r3(x.jaccard),spearman_shared=r3(x.spearman_shared_only),kendall_shared=r3(x.kendall_shared_only),spearman_union=r3(x.spearman_union_absent_last),kendall_union=r3(x.kendall_union_absent_last))
            d[key]=v
        return d
    gini=pd.read_csv(r/'06_socio_epistemic/titleword_concentration_gini.csv')
    gini_out={int(x.year):{'vocab':int(x.vocabulary_size),'gini_full':r3(x.gini_full_positive_vocabulary),'gini_top20':r3(x.gini_top20_renormalized),'top20_mass':r3(x.top20_mass)} for _,x in gini.iterrows()}
    pre=pd.read_csv(r/'15_titleword_preprocessing_sensitivity/balanced_profile_sensitivity_vs_main.csv')
    pre_out={'min_top20_overlap_pct':{},'max_js':{}}
    for y,g in pre.groupby('year'):
        pre_out['min_top20_overlap_pct'][int(y)]=r1(pct(g.top_n_overlap_rate.min()))
        pre_out['max_js'][int(y)]=r4(g.js_divergence_full_profiles.max())
    alloc=pd.read_csv(r/'06_socio_epistemic/titleword_community_allocation_sensitivity.csv')
    alloc_out={'top20_overlap_pct':{int(x.year):r1(pct(x.top_n_overlap_rate)) for _,x in alloc.iterrows()},'js':{int(x.year):r4(x.js_divergence_full_profiles) for _,x in alloc.iterrows()}}

    surv=pd.read_csv(r/'06_socio_epistemic/author_survival_retention.csv')
    continuity={}
    for e,l in [(1979,1986),(1986,1996),(1996,2001)]:
        x=surv[(surv.earlier_year==e)&(surv.later_year==l)&(surv.population=='LCC')].iloc[0]
        w=surv[(surv.earlier_year==e)&(surv.later_year==l)&(surv.population=='whole_year_network')].iloc[0]
        continuity[f'{e}-{l}']={'earlier_lcc':int(x.earlier_authors),'later_lcc':int(x.later_authors),'shared':int(x.shared_authors),'survival_pct':r2(pct(x.survival_rate)),'retention_pct':r2(pct(x.retention_rate)),'annualized_survival':r3(x.annualized_survival_equivalent),'whole_network_survival_pct':r2(pct(w.survival_rate))}
    lt=pd.read_csv(r/'06_socio_epistemic/long_term_author_degree_position.csv')
    continuity['persistent_all_four_lccs']=int(lt.author.nunique())
    continuity['median_lcc_degree_percentiles']=[r3(lt[lt.year==y].degree_percentile_in_lcc.median()) for y in [1979,1986,1996,2001]]

    trend=pd.read_csv(r/'11_publication_trends/common_relative_trend_periods.csv')
    increases=[[int(x.start_year),int(x.end_year)] for _,x in trend[trend.direction.eq('increase')].iterrows()]
    decreases=[[int(x.start_year),int(x.end_year)] for _,x in trend[trend.direction.eq('decrease')].iterrows()]
    def coverage(year, title=None):
        g=p3[p3.year.eq(year)]; counts=g.source_title.fillna('').value_counts(); share=counts.iloc[0]/len(g)*100
        if title is not None: share=(g.source_title.fillna('').str.upper()==title).sum()/len(g)*100
        return int(len(g)),r1(share)
    c93=coverage(1993); c94=coverage(1994,'PHYSICS OF PLASMAS')
    h1=pd.read_csv(r/'10_historical_title_contrasts/historical_title_contrast_1968-1974_vs_1956-1967.csv')
    h2=pd.read_csv(r/'10_historical_title_contrasts/historical_title_contrast_1994-1996_vs_1989-1993.csv')
    leading=h2[~h2.feature.astype(str).str.contains(' ')].head(9).feature.tolist(); leading=[('JT-60U' if x.lower()=='jt-60u' else ('Alfven' if x.lower()=='alfven' else x)) for x in leading]

    hist={}
    for key in ['1959-1968','1969-1973','1974-1978','1983','1986']:
        x=snapmap[key]; out_key=int(key) if key in {'1983','1986'} else key; hist[out_key]={'papers':int(x.papers),'authors':int(x.number_of_nodes),'lcc_authors':int(x.largest_component_size),'lcc_edges':int(x.largest_component_edges),'lcc_share_pct':r1(pct(x.ratio_largest_component_nodes))}
    x=snapmap['1974-1978']; hist['1974-1978_full_network_edges']=int(x.number_of_edges); hist['1974-1978_lcc_edge_share_pct']=r1(pct(x.ratio_largest_component_edges))

    # Authorship-scale aggregates are direct grouped summaries of the saved analytical corpus.
    def period_stats(lo,hi):
        g=a[a.year.between(lo,hi)]; n=g.n_authors_network.astype(float)
        return r1(pct((n==1).mean())),r2(n.mean())
    s1,m1=period_stats(1956,1968); s2,m2=period_stats(1996,2007)
    y86=a[a.year.eq(1986)].n_authors_network.astype(int)

    icr=pd.read_csv(r/'05_internationalization/yearly_ICR_geographic.csv')
    icr_out={}
    for y in [1973,1979,1983,1986,1991,1996,2000,2004,2007]:
        x=icr[icr.year.eq(y)].iloc[0]; icr_out[y]={'analytical_publications':int(x.total_papers),'country_coded':int(x.papers_with_country),'coverage_pct':r1(pct(x.coverage_rate)),'international':int(x.international_papers),'icr_pct':r1(pct(x.ICR))}

    coc={}
    for period in ['1979-1982','1986-1987','1991-1995','1996-2000']:
        d=pd.read_csv(r/f'05_internationalization/cocountry_{period}_lcc_nodes.csv').sort_values('betweenness',ascending=False).head(5)
        coc[period]=[[x.actor,int(x.degree),int(round(x.weighted_degree)),r3(x.betweenness)] for _,x in d.iterrows()]
    bench=pd.read_csv(r/'05_internationalization/cocountry_1979-1982_JET_geographic_benchmark.csv').set_index('actor')
    bench_out={'geographic_uk_betweenness':r3(bench.loc['UK','geographic_betweenness']),'modified_uk_betweenness':r3(bench.loc['UK','modified_betweenness']),'geographic_usa_betweenness':r3(bench.loc['USA','geographic_betweenness']),'modified_usa_betweenness':r3(bench.loc['USA','modified_betweenness']),'geographic_germany_betweenness':r3(bench.loc['GERMANY','geographic_betweenness']),'modified_germany_betweenness':r3(bench.loc['GERMANY','modified_betweenness'])}
    eu=pd.read_csv(r/'05_internationalization/european_indicators_principal_periods.csv')
    eu_out={'columns':['coverage_pct','icr_pct','european_fractional_output_share_pct','conditional_european_partner_share_pct','oe_affinity']}
    for _,x in eu.iterrows():
        lo,hi=map(int,str(x.period).split('-')); ig=icr[icr.year.between(lo,hi)]
        period_icr=ig.international_papers.sum()/ig.papers_with_country.sum()
        eu_out[x.period]=[r1(pct(x.coverage)),r1(pct(period_icr)),r1(pct(x.european_fractional_output_share)),r1(pct(x.conditional_european_partner_share)),r3(x.oe_european_affinity)]
    en=pd.read_csv(r/'05_internationalization/european_enlargement_definition_sensitivity.csv')
    en_out={}
    labels={1986:('EC-12','EC-10'),1995:('EU-15','EC-12'),2004:('EU-25','EU-15')}
    for y in [1986,1995,2004]:
        g=en[en.year.eq(y)].set_index('definition'); en_out[y]={'dynamic_set':labels[y][0],'legacy_core':labels[y][1],
          'output_share_dynamic_core_pct':[r1(pct(g.loc['dynamic_set','european_fractional_output_share'])),r1(pct(g.loc['legacy_core','european_fractional_output_share']))],
          'partner_share_dynamic_core_pct':[r1(pct(g.loc['dynamic_set','conditional_european_partner_share'])),r1(pct(g.loc['legacy_core','conditional_european_partner_share']))],
          'affinity_dynamic_core':[r3(g.loc['dynamic_set','oe_european_affinity']),r3(g.loc['legacy_core','oe_european_affinity'])]}

    histseed=pd.read_csv(r/'06b_louvain_seed_robustness/historical_lcc_louvain_seed_sensitivity.csv')
    hs={}
    for period,g in histseed.groupby('period'):
        key=1986 if str(period) in {'1986','1986.0'} else str(period)
        hs[key]={'median_ari':r3(g.adjusted_rand_index.median()),'min_ari':r3(g.adjusted_rand_index.min()),'modularity_range':[r4(g.modularity.min()),r4(g.modularity.max())]}

    # Direct country reconstruction coverage from saved geographic lists.
    has_country=a.countries_normalized.apply(lambda x: bool(parse_list_cell(x)))
    cov_early={}
    for label,lo,hi in [('1956-1968_reconstructable_country_pct',1956,1968),('1969-1972_reconstructable_country_pct',1969,1972)]:
        mask=a.year.between(lo,hi); cov_early[label]=r1(pct(has_country[mask].mean()))
    def rp_jet_prefix_audit(frame):
        raw=frame.reprint_address_one_line.fillna('').astype(str)
        cleaned=raw.map(strip_rp_author_prefix)
        # The frozen audit counts only explicit JET Joint Undertaking strings
        # whose JET status was actually hidden by the RP author prefix.  Six
        # prefixed strings were already visible to the legacy Culham/EFDA guard.
        legacy_visible=(raw.str.contains(r'(?i)\b(?:CULHAM|EFDA)\b',regex=True)
                        & raw.str.contains(r'(?i)\bJET\b',regex=True))
        blocked=(cleaned.ne(raw.str.strip())
                 & cleaned.str.startswith(('JET JOINT UNDERTAKING','Jet Joint Undertaking'))
                 & ~legacy_visible)
        return int(blocked.sum())

    result={
      'shared':{
        'preserved_retrieval_corpus':{'records':len(p3),'unique_ut':int(p3.ut.nunique()),'columns':len(p3.columns),'sha256':hashlib.sha256(preserved.read_bytes()).hexdigest()},
        'analytical_corpus':{'records':cs['corpus']['analytical_records'],'exclusions_total':cs['corpus']['exclusions_total'],'editorial_material_excluded':cs['corpus']['editorial_material_excluded'],'additional_non_editorial_event_summaries_excluded':cs['corpus']['additional_non_editorial_excluded']},
        'author_identity':{'normalized_forms':p3s['distinct_normalized_author_forms'],'canonical_identities':p3s['canonical_author_nodes'],'merged_variants':p3s['variant_forms_merged'],'automatic_merges':p3s['auto_merges_applied'],'curated_merge':ids['override_merge_rows'],'curated_keep_separate':ids['override_keep_separate_rows'],'analytical_normalized_forms':cs['identity_subset']['analytical_normalized_forms'],'analytical_canonical_identities':cs['identity_subset']['analytical_canonical_identities'],'same_paper_identity_collisions':ids['same_paper_identity_collisions']}
      },
      'guzzardi_lalli':{
        'corpus_routing':{'publication_counts':'preserved_retrieval','relative_publication_trends':'preserved_retrieval','networks':'analytical','communities':'analytical','titleword_snapshots':'analytical','historical_title_contrasts':'analytical','author_continuity':'analytical'},
        'network_snapshots':{y:snapshot(y) for y in [1979,1986,1996,2001]},
        'annual_lcc':{'first_year_above_50_pct':1985,'first_year_above_50_value_pct':r1(pct(annualmap[1985].ratio_largest_component_nodes))},
        'team_size_sensitivity_lcc_pct':{k:v for k,v in team_sens.items() if k!=1983},
        'document_type_sensitivity_lcc_pct':{k:v for k,v in doc_sens.items() if k!=1983},
        'louvain':{'manuscript_seed':42,'sensitivity_seeds':[1,7,21,42,84],'resolution':1.0,'edge_weight':'paper_count','seed_sensitivity':seed_out},
        'lcc_representativeness':rep_out,'cross_snapshot_lexical_recurrence_community_balanced':sim_out('community_balanced'),
        'cross_snapshot_lexical_recurrence_paper_pooled':sim_out('paper_pooled_lcc',False),'titleword_concentration':gini_out,
        'preprocessing_sensitivity':pre_out,'allocation_sensitivity_author_proportional_vs_equal':alloc_out,'author_continuity':continuity,
        'publication_trends':{'smoother_years':2,'minimum_run_years':3,'minimum_mean_absolute_annual_change':0.04,'excluded_detection_years':[1962],'increase_intervals':increases,'decrease_intervals':decreases,'source_coverage_1993':{'records':c93[0],'leading_source_share_pct':c93[1]},'source_coverage_1994':{'records':c94[0],'physics_of_plasmas_share_pct':c94[1]}},
        'historical_title_contrasts':{'1968-1974_vs_1956-1967':{'required_highest_ranked_feature':str(h1.iloc[0].feature)},'1994-1996_vs_1989-1993':{'required_leading_sequence':leading}}
      },
      'lalli':{
        'historical_author_networks':hist,'annual_lcc_selected':{y:r1(pct(annualmap[y].ratio_largest_component_nodes)) for y in [1985,1986,1987,1988,1989]},
        'authorship_scale':{'single_authored_share_pct':{'1956-1968':s1,'1996-2007':s2},'mean_team_size':{'1956-1968':m1,'1996-2007':m2},'1986_large_team_counts':{'papers_gt20':int((y86>20).sum()),'papers_gt50':int((y86>50).sum()),'max_team_size':int(y86.max())}},
        'team_size_sensitivity_lcc_pct':team_sens,'document_type_sensitivity_lcc_pct':doc_sens,
        'louvain_seed_sensitivity':{'manuscript_seed':42,'alternative_seeds':'0-19',**hs},
        'address_coverage':{1973:r1(pct(icr[icr.year.eq(1973)].iloc[0].coverage_rate)),1974:r1(pct(icr[icr.year.eq(1974)].iloc[0].coverage_rate)),1975:r1(pct(icr[icr.year.eq(1975)].iloc[0].coverage_rate)),**cov_early},
        'rp_jet_parser_audit':{'rp_strings_with_prefix_initially_blocking_jet':rp_jet_prefix_audit(p3),'analytical_corpus_count':rp_jet_prefix_audit(a)},
        'icr_selected_years':icr_out,'modified_cocountry_top_betweenness':coc,'jet_geographic_benchmark_1979_1982':bench_out,
        'european_period_measures':eu_out,'enlargement_sensitivity':en_out
      }
    }
    Path(args.output).write_text(yaml.safe_dump(result,sort_keys=False,allow_unicode=True),encoding='utf-8')
    print(args.output)

if __name__=='__main__': main()
