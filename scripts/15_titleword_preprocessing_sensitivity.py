#!/usr/bin/env python3
"""Sensitivity audit for the Guzzardi-Lalli titleword preprocessing.

Compares the submission profile (frozen English stopwords + `fusion`) against:
1) frozen English stopwords only;
2) English + `fusion`, `nuclear`, `thermonuclear`;
3) the submission stopwords while preserving ordinal tokens.

All profiles share markup stripping, NFKC/lowercase normalization, pure-number
removal, hyphen-preserving tokenization, and the frozen inflectional merge map.
"""
from pathlib import Path
import argparse, json, sys
import pandas as pd

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
from fusion_pipeline.networks import prepare_network_dataframe
from fusion_pipeline.socio_epistemic import (
    load_titleword_config, load_titleword_stopword_profiles,
    community_titleword_profiles, compare_top_lists, js_divergence,
    tokenize_title, snapshot_social_structure,
)
from fusion_pipeline.community_vocabulary import community_vocabulary_figure_data, plot_community_vocabulary_panels


def selected_terms_by_community(data: pd.DataFrame) -> dict[int,set[str]]:
    out={}
    if data.empty:
        return out
    for c,g in data[data['selected_term']].groupby('community'):
        out[int(c)]=set(g['term'].astype(str))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase3_csv')
    ap.add_argument('--output-dir',default='outputs_titleword_preprocessing_sensitivity')
    ap.add_argument('--config-dir',default=str(ROOT/'config'))
    ap.add_argument('--years',nargs='*',type=int,default=[1979,1986,1996,2001])
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--top-communities',type=int,default=3)
    ap.add_argument('--terms-per-community',type=int,default=5)
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=prepare_network_dataframe(pd.read_csv(args.phase3_csv,low_memory=False))
    _,merge_map=load_titleword_config(args.config_dir)
    stop_profiles=load_titleword_stopword_profiles(args.config_dir)
    variants={
        'main': (stop_profiles['main'], True),
        'english_only': (stop_profiles['english_only'], True),
        'expanded_domain': (stop_profiles['expanded_domain'], True),
        'main_preserve_ordinals': (stop_profiles['main'], False),
    }

    profiles={name:{} for name in variants}
    figdata={name:{} for name in variants}
    social={year:snapshot_social_structure(df,year,seed=args.seed) for year in args.years}
    summary_rows=[]
    for name,(stopset,remove_ordinals) in variants.items():
        vdir=out/name; vdir.mkdir(exist_ok=True)
        panels={}; allrows=[]
        for year in args.years:
            r=community_titleword_profiles(
                df,year,stopset,merge_map,seed=args.seed,
                remove_ordinals=remove_ordinals, social=social[year],
            )
            profiles[name][year]=r['balanced_profile']
            d=community_vocabulary_figure_data(r,year,args.top_communities,args.terms_per_community)
            d.to_csv(vdir/f'{year}_figure5_data.csv',index=False)
            panels[year]=d; figdata[name][year]=d; allrows.append(d)
            summary_rows.append({
                'variant':name,'year':year,
                'n_communities':int(r['community_summary'].shape[0]),
                'balanced_vocabulary_size':len(r['balanced_profile']),
                'edge_support_papers':int(r['edge_support_papers']),
            })
        combined=pd.concat(allrows,ignore_index=True)
        combined.to_csv(vdir/'Figure_5_community_vocabulary_data.csv',index=False)
        plot_community_vocabulary_panels(panels,vdir/'Figure_5_community_vocabulary.png',args.years,dpi=300)

    # Profile-level sensitivity relative to submission/main.
    rows=[]
    for name in variants:
        if name=='main':
            continue
        for year in args.years:
            row={'variant':name,'year':year}
            row.update(compare_top_lists(profiles['main'][year],profiles[name][year],20))
            row['js_divergence_full_profiles']=js_divergence(profiles['main'][year],profiles[name][year])
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/'balanced_profile_sensitivity_vs_main.csv',index=False)

    # Figure-5 term-selection stability for the same social communities.
    term_rows=[]
    for name in variants:
        if name=='main':
            continue
        for year in args.years:
            a=selected_terms_by_community(figdata['main'][year])
            b=selected_terms_by_community(figdata[name][year])
            for c in sorted(set(a)|set(b)):
                A=a.get(c,set()); B=b.get(c,set()); U=A|B; I=A&B
                term_rows.append({
                    'variant':name,'year':year,'community':c,
                    'main_n_terms':len(A),'variant_n_terms':len(B),
                    'overlap_count':len(I),
                    'overlap_rate_over_main_top5':len(I)/len(A) if A else float('nan'),
                    'jaccard':len(I)/len(U) if U else float('nan'),
                    'main_terms':'|'.join(sorted(A)),
                    'variant_terms':'|'.join(sorted(B)),
                })
    pd.DataFrame(term_rows).to_csv(out/'figure5_term_selection_sensitivity_vs_main.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(out/'variant_summary.csv',index=False)

    # Explicit artefact audit for tags and ordinal rules on the complete corpus.
    main_stop=stop_profiles['main']
    token_counts={'sub':0,'sup':0}
    ordinal_retained=0
    for title in df['title'].fillna(''):
        toks=tokenize_title(title,main_stop,merge_map,remove_ordinals=True)
        token_counts['sub'] += toks.count('sub')
        token_counts['sup'] += toks.count('sup')
        # A defensive check: no bare ordinal token should survive the main rule.
        ordinal_retained += sum(1 for t in toks if __import__('re').fullmatch(r'\d+(?:st|nd|rd|th)',t))
    audit={
        'main_profile': 'frozen English stopwords + fusion',
        'english_stopword_count': len(stop_profiles['english_only']),
        'main_stopword_count': len(stop_profiles['main']),
        'expanded_domain_stopword_count': len(stop_profiles['expanded_domain']),
        'markup_artifact_tokens_after_cleaning': token_counts,
        'bare_ordinal_tokens_after_main_cleaning': ordinal_retained,
        'domain_stopwords_main': sorted(stop_profiles['main']-stop_profiles['english_only']),
        'domain_stopwords_expanded': sorted(stop_profiles['expanded_domain']-stop_profiles['english_only']),
    }
    (out/'preprocessing_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    main()
