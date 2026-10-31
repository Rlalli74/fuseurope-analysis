#!/usr/bin/env python3
"""Canonical social/titleword analyses for 1979, 1986, 1996, 2001."""
from pathlib import Path
from collections import Counter
import argparse, json, sys
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.networks import prepare_network_dataframe
from fusion_pipeline.socio_epistemic import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase3_csv');ap.add_argument('--output-dir',default='outputs_phase6');
    ap.add_argument('--config-dir',default=str(ROOT/'config'));ap.add_argument('--years',nargs='*',type=int,default=[1979,1986,1996,2001]);args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    df=prepare_network_dataframe(pd.read_csv(args.phase3_csv,low_memory=False))
    exclusions,merge_map=load_titleword_config(args.config_dir)
    stopword_profiles=load_titleword_stopword_profiles(args.config_dir)
    snapshots=sorted(dict.fromkeys(args.years))
    if len(snapshots) < 2: raise SystemExit("Phase 06 requires at least two snapshot years")
    results={}; profiles={}; sensitivity_profiles={name:{} for name in stopword_profiles}
    social={y:snapshot_social_structure(df,y,seed=42) for y in snapshots}
    concentration=[]; represent=[]; top_tables=[]; pooled_counts_all=Counter(); allocation_sensitivity=[]; weighting_profiles={'community_balanced':{},'paper_pooled_lcc':{}}
    for y in snapshots:
        r=community_titleword_profiles(df,y,exclusions,merge_map,social=social[y],allocation_mode='equal_communities');results[y]=r;profiles[y]=r['balanced_profile'];pooled_counts_all.update(r['pooled_lcc_counts'])
        weighting_profiles['community_balanced'][y]=r['balanced_profile']; weighting_profiles['paper_pooled_lcc'][y]=r['pooled_lcc_profile']
        r_author=community_titleword_profiles(df,y,exclusions,merge_map,social=social[y],allocation_mode='author_proportional')
        aq={'year':y}; aq.update(compare_top_lists(r['balanced_profile'],r_author['balanced_profile'],20)); aq.update(rank_correlations(r['balanced_profile'],r_author['balanced_profile'],20)); aq['js_divergence_full_profiles']=js_divergence(r['balanced_profile'],r_author['balanced_profile']); allocation_sensitivity.append(aq)
        for pname,pset in stopword_profiles.items():
            rr=r if pname=='main' else community_titleword_profiles(df,y,pset,merge_map,social=social[y])
            sensitivity_profiles[pname][y]=rr['balanced_profile']
        r['community_summary'].to_csv(out/f'{y}_community_summary.csv',index=False)
        r['paper_assignments'].to_csv(out/f'{y}_community_paper_assignments.csv',index=False)
        community_terms_table(r,25).to_csv(out/f'{y}_community_titlewords_top25.csv',index=False)
        community_tfidf_table(r,25).to_csv(out/f'{y}_community_tfidf_top25.csv',index=False)
        for ptype,prof in [('community_balanced',r['balanced_profile']),('paper_pooled_lcc',r['pooled_lcc_profile']),('whole_field',r['whole_field_profile'])]:
            top_tables.append(profile_table(prof,y,ptype,20))
        c=vocabulary_concentration(r['balanced_profile'],20);c['year']=y;concentration.append(c)
        q=lcc_whole_field_comparison(r,20,lcc_profile='paper_pooled');q['year']=y;q['lcc_edge_support_papers']=r['edge_support_papers'];q['total_papers']=len(r['sub']);represent.append(q)
    pd.concat(top_tables,ignore_index=True).to_csv(out/'titleword_top20_profiles.csv',index=False)
    pd.DataFrame(concentration).to_csv(out/'titleword_concentration_gini.csv',index=False)
    pd.DataFrame(represent).to_csv(out/'lcc_vs_whole_field_titleword_comparison.csv',index=False)
    pd.DataFrame(allocation_sensitivity).to_csv(out/'titleword_community_allocation_sensitivity.csv',index=False)

    pair_rows=[]; sensitivity_rows=[]
    for pname,pp in sensitivity_profiles.items():
        for a,b in zip(snapshots,snapshots[1:]):
            row={'stopword_profile':pname,'earlier_year':a,'later_year':b}
            row.update(compare_top_lists(pp[a],pp[b],20))
            row.update(rank_correlations(pp[a],pp[b],20))
            row['js_divergence_full_profiles']=js_divergence(pp[a],pp[b])
            sensitivity_rows.append(row)
            if pname=='main': pair_rows.append(dict(row))
    pd.DataFrame(pair_rows).to_csv(out/'titleword_snapshot_similarity.csv',index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out/'titleword_snapshot_similarity_stopword_sensitivity.csv',index=False)
    weighting_rows=[]
    for ptype,pp in weighting_profiles.items():
        for a,b in zip(snapshots,snapshots[1:]):
            row={'profile_type':ptype,'earlier_year':a,'later_year':b}
            row.update(compare_top_lists(pp[a],pp[b],20)); row.update(rank_correlations(pp[a],pp[b],20)); row['js_divergence_full_profiles']=js_divergence(pp[a],pp[b]); weighting_rows.append(row)
    pd.DataFrame(weighting_rows).to_csv(out/'titleword_snapshot_similarity_weighting_sensitivity.csv',index=False)

    equal=aggregate_profiles(profiles); pooled=normalize_counter(pooled_counts_all)
    eq=profile_table(equal,'1979-2001','equal_snapshot_aggregate',None);po=profile_table(pooled,'1979-2001','paper_pooled_aggregate',None)
    eq.head(50).to_csv(out/'titleword_equal_snapshot_aggregate_top50.csv',index=False);po.head(50).to_csv(out/'titleword_paper_pooled_aggregate_top50.csv',index=False)
    comp=pd.merge(eq[['term','weight']].rename(columns={'weight':'equal_snapshot_weight'}),po[['term','weight']].rename(columns={'weight':'paper_pooled_weight'}),on='term',how='outer').fillna(0)
    comp.to_csv(out/'titleword_aggregate_weight_comparison.csv',index=False)

    lcc_sets=snapshot_author_sets(df,snapshots,'authors_network',True); whole_sets=snapshot_author_sets(df,snapshots,'authors_network',False)
    turn_lcc=turnover_table(lcc_sets);turn_lcc['population']='LCC';turn_whole=turnover_table(whole_sets);turn_whole['population']='whole_year_network'
    pd.concat([turn_lcc,turn_whole],ignore_index=True).to_csv(out/'author_survival_retention.csv',index=False)
    # Identity-resolution robustness: standardized names vs curated disambiguated names.
    norm_sets=snapshot_author_sets(df,snapshots,'authors_normalized',True);turn_norm=turnover_table(norm_sets);turn_norm['population']='LCC_normalized_names_pre_disambiguation';turn_norm.to_csv(out/'author_turnover_name_disambiguation_sensitivity.csv',index=False)
    common,pos=long_term_author_position(df,snapshots);pos.to_csv(out/'long_term_author_degree_position.csv',index=False)
    pd.DataFrame({'author':sorted(common)}).to_csv(out/'long_term_authors_all_four_lccs.csv',index=False)

    summary={'snapshots':snapshots,'titleword_unit':'token occurrences in edge-supporting LCC papers; principal allocation gives each represented community 1/q, with author-proportional allocation reported as sensitivity',
             'primary_LCC_profile':'equal mean of within-community normalized titleword distributions',
             'robustness_profiles':['direct paper-pooled LCC profile','paper-pooled LCC vs paper-pooled whole-field representativeness','author-proportional community allocation','community TF-IDF','English-only stopwords','expanded domain stopwords (fusion, nuclear, thermonuclear)'],
             'titleword_preprocessing':'frozen English stopwords + corpus-definition stopword fusion; NFKC/lowercase; markup stripped; pure numerals and ordinal markers removed; hyphenated compounds preserved; conservative plural merge map',
             'top20_overlap':'symmetric intersection/20; Jaccard also reported','rank_correlations':'shared-only and union with absent terms tied at last rank, both labelled explicitly',
             'gini':'reported both for full positive vocabulary and renormalized displayed top-20',
             'survival':'shared/earlier cohort','retention':'shared/later cohort; unequal interval length reported and annualized survival equivalent S^(1/g) supplied',
             'long_term_authors':len(common)}
    (out/'phase6_analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
