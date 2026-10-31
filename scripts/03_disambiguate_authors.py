#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))

from fusion_pipeline.disambiguation import run_disambiguation_from_phase2_dataframe, serialize_list, candidate_context
from fusion_pipeline.phase3_validation import (
    add_review_impact, compare_with_legacy_auto_merges, disambiguation_summary,
    network_sensitivity, historical_manual_review_shortlist,
)

LIST_COLS=[
    'authors','authors_wos','authors_full_wos','document_types','keywords','keywords_plus',
    'affiliations','institutions','institutions_normalized','country_address_tails','wos_categories','research_areas',
    'authors_normalized','author_normalization_status','countries_normalized','countries_network',
    'authors_disambiguated','authors_network',
]

def records_to_df(records):
    df=pd.DataFrame(records)
    for c in LIST_COLS:
        if c in df.columns:
            df[c]=df[c].apply(lambda x:serialize_list(x) if isinstance(x,list) else x)
    return df


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase2_csv',help='preserved_corpus_phase2.csv from script 02')
    ap.add_argument('--manual-overrides',default=str(ROOT/'corrections'/'author_identity_overrides.csv'))
    ap.add_argument('--legacy-analysis-dir',default=None,help='Optional old analysis folder for audit comparison')
    ap.add_argument('--output-dir',default='outputs_phase3')
    args=ap.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)

    phase2=pd.read_csv(args.phase2_csv,low_memory=False)
    res=run_disambiguation_from_phase2_dataframe(phase2,args.manual_overrides)
    records=res['records']; candidates=res['candidates']; variant=res['variant_map']; log=res['merge_log']

    review=add_review_impact(candidates,records)
    historical_shortlist=historical_manual_review_shortlist(candidates,records,max_recommended=30,manual_overrides=res['manual_overrides'])
    context_rows=[]
    if not historical_shortlist.empty:
        for _,hr in historical_shortlist[historical_shortlist['recommended_for_manual_check']==1].iterrows():
            ctx=candidate_context(records,hr['author_a'],hr['author_b'])
            if ctx.empty:
                continue
            ctx.insert(0,'manual_review_rank',int(hr['manual_review_rank']))
            ctx.insert(1,'author_a',hr['author_a']); ctx.insert(2,'author_b',hr['author_b'])
            ctx=ctx[(pd.to_numeric(ctx['year'],errors='coerce')>=1959)&(pd.to_numeric(ctx['year'],errors='coerce')<=1986)]
            context_rows.append(ctx)
    historical_context=pd.concat(context_rows,ignore_index=True) if context_rows else pd.DataFrame()
    sens=network_sensitivity(records)
    legacy_summary,legacy=compare_with_legacy_auto_merges(candidates,args.legacy_analysis_dir)

    records_to_df(records).to_csv(out/'preserved_identity_corpus_phase3.csv',index=False)
    candidates.to_csv(out/'author_candidate_pairs.csv',index=False)
    variant.to_csv(out/'author_variant_map_phase3.csv',index=False)
    log.to_csv(out/'author_merge_log.csv',index=False)
    review.to_csv(out/'author_review_priority.csv',index=False)
    historical_shortlist.to_csv(out/'author_manual_review_shortlist_lalli.csv',index=False)
    if not historical_context.empty: historical_context.to_csv(out/'author_manual_review_context_lalli_top30.csv',index=False)
    sens.to_csv(out/'author_disambiguation_network_sensitivity.csv',index=False)
    if not legacy.empty: legacy.to_csv(out/'legacy_auto_merge_comparison.csv',index=False)

    summary=disambiguation_summary(variant,candidates,log,review)
    summary['lalli_historical_manual_shortlist']={
        'shortlist_rows':int(len(historical_shortlist)),
        'recommended_for_manual_check':int(historical_shortlist['recommended_for_manual_check'].sum()) if not historical_shortlist.empty else 0,
    }
    summary['legacy_comparison']=legacy_summary
    with open(out/'phase3_validation_summary.json','w',encoding='utf-8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
