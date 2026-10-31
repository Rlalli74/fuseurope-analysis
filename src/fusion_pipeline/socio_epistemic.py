"""Canonical social/textual analyses for the Guzzardi-Lalli article.

The implementation makes the unit of analysis explicit. A publication can link
several Louvain communities; its titleword occurrences are divided fractionally
across the distinct communities represented by its LCC authors. Community
profiles are normalized internally, and the primary LCC profile is the equal-
community mean. A direct paper-pooled profile is produced as a robustness check.

"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import math, re, unicodedata

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.proportion import proportion_confint

from .networks import (
    prepare_network_dataframe, filter_publications, build_coauthorship_graph,
    largest_component_graph, louvain_partition, parse_list_cell,
)

SNAPSHOTS=(1979,1986,1996,2001)
TOKEN_RE=re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
MARKUP_RE=re.compile(r"<[^>]+>")
ORDINAL_TOKEN_RE=re.compile(r"^(\d+)(?:st|nd|rd|th)(?:-(.+))?$")
BARE_ORDINAL_RE=re.compile(r"^\d+(?:st|nd|rd|th)$")
DANGLING_PREFIX_RE=re.compile(r"\b[A-Za-z]{2,}-\s+(?=(?:and|or)\b)")


def _load_word_set(path):
    out=set()
    path=Path(path)
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        s=line.strip().lower()
        if s and not s.startswith('#'):
            out.add(s)
    return out


def load_titleword_config(config_dir, domain_stopwords_file='domain_stopwords_main.txt'):
    """Load the frozen titleword preprocessing specification.

    The default submission profile combines the repository-frozen standard
    English stop-word list, any explicitly documented additional exclusions,
    and the corpus-definition domain stopwords (currently only ``fusion``).
    Passing ``domain_stopwords_file=None`` returns the English-only profile.
    """
    config_dir=Path(config_dir)
    exclusions=_load_word_set(config_dir/'english_stopwords.txt')
    exclusions |= _load_word_set(config_dir/'titleword_exclusions.txt')
    if domain_stopwords_file:
        exclusions |= _load_word_set(config_dir/domain_stopwords_file)
    m=pd.read_csv(config_dir/'titleword_merge_map.csv')
    merge={str(r.variant).strip().lower():str(r.canonical).strip().lower() for _,r in m.iterrows()}
    return exclusions,merge


def load_titleword_stopword_profiles(config_dir):
    """Return the three pre-registered stopword profiles used in sensitivity checks."""
    config_dir=Path(config_dir)
    english=_load_word_set(config_dir/'english_stopwords.txt') | _load_word_set(config_dir/'titleword_exclusions.txt')
    main=english | _load_word_set(config_dir/'domain_stopwords_main.txt')
    expanded=english | _load_word_set(config_dir/'domain_stopwords_robustness_expanded.txt')
    return {'english_only':english,'main':main,'expanded_domain':expanded}


def tokenize_title(text, exclusions=None, merge_map=None, extra_exclusions=None, remove_ordinals=True):
    """Tokenize a WoS title under explicit, reproducible cleaning rules.

    Rules: NFKC + lowercase; strip embedded HTML/XML presentation tags while
    retaining their contents; preserve hyphenated technical compounds; remove
    pure numerals; optionally remove ordinal markers (``2nd`` -> dropped,
    ``2nd-harmonic`` -> ``harmonic``); then apply the frozen stopword set and
    the conservative inflectional merge map.
    """
    exclusions=set(exclusions or ()) | set(extra_exclusions or ())
    merge_map=merge_map or {}
    s=unicodedata.normalize('NFKC', '' if text is None else str(text)).lower()
    # WoS titles can contain markup such as Li<sub>2</sub>O or Z<sub>eff</sub>.
    # Removing tags before tokenization prevents the artefactual tokens 'sub'/'sup'.
    s=MARKUP_RE.sub('',s)
    # Drop incomplete coordinated prefixes such as 'sub- and supersonic':
    # a dangling prefix is not a usable unigram and should not become 'sub'.
    s=DANGLING_PREFIX_RE.sub('',s)
    toks=[]
    for t in TOKEN_RE.findall(s):
        if t.isdigit():
            continue
        if remove_ordinals:
            m=ORDINAL_TOKEN_RE.fullmatch(t)
            if m:
                if not m.group(2) or BARE_ORDINAL_RE.fullmatch(m.group(2)):
                    continue
                t=m.group(2)
        if len(t)<2 or t in exclusions:
            continue
        t=merge_map.get(t,t)
        if t not in exclusions:
            toks.append(t)
    return toks


def normalize_counter(c):
    total=float(sum(c.values()))
    return {k:float(v)/total for k,v in c.items()} if total else {}


def snapshot_social_structure(df, year, seed=42, authors_col='authors_network'):
    sub=filter_publications(df,year,year)
    G=build_coauthorship_graph(sub,authors_col=authors_col)
    L=largest_component_graph(G)
    part=louvain_partition(L,seed=seed,weight='paper_count')
    return sub,G,L,part


def community_titleword_profiles(df, year, exclusions, merge_map, seed=42, extra_exclusions=None, remove_ordinals=True, social=None, allocation_mode='equal_communities'):
    """Return community, community-balanced, pooled-LCC and whole-field profiles.

    LCC title papers are records containing >=2 distinct LCC authors, i.e. papers
    that actually support at least one co-authorship edge inside the LCC.
    In the principal ``equal_communities`` allocation, if a paper includes
    authors from q Louvain communities, every titleword occurrence contributes
    weight 1/q to each represented community.  ``author_proportional`` instead
    allocates each occurrence in proportion to the number of distinct LCC
    authors from each represented community on that paper.
    """
    sub,G,L,part=social if social is not None else snapshot_social_structure(df,year,seed=seed)
    lcc=set(L.nodes())
    comm_counts=defaultdict(Counter); comm_papers=Counter(); pooled=Counter(); whole=Counter()
    edge_support_papers=0
    paper_rows=[]
    for idx,r in sub.iterrows():
        toks=tokenize_title(r.get('title',''),exclusions,merge_map,extra_exclusions,remove_ordinals=remove_ordinals)
        whole.update(toks)
        authors=list(dict.fromkeys(a for a in r['authors_network'] if a in lcc))
        if len(authors)<2: continue
        community_author_counts=Counter(part[a] for a in authors if a in part)
        communities=sorted(community_author_counts)
        if not communities: continue
        edge_support_papers+=1; pooled.update(toks)
        if allocation_mode == 'equal_communities':
            weights={c:1.0/len(communities) for c in communities}
        elif allocation_mode == 'author_proportional':
            denom=float(sum(community_author_counts.values()))
            weights={c:community_author_counts[c]/denom for c in communities}
        else:
            raise ValueError(f'Unknown community titleword allocation mode: {allocation_mode}')
        for c in communities:
            comm_papers[c]+=1
            for t in toks: comm_counts[c][t]+=weights[c]
        paper_rows.append({'record_id':r.get('record_id',''),'year':year,
                           'n_lcc_authors':len(authors),'n_communities':len(communities),
                           'communities':'|'.join(map(str,communities)),
                           'allocation_mode':allocation_mode,
                           'community_weights':'|'.join(f'{c}:{weights[c]:.12g}' for c in communities)})
    comm_profiles={c:normalize_counter(cnt) for c,cnt in comm_counts.items()}
    vocab=set().union(*(p.keys() for p in comm_profiles.values())) if comm_profiles else set()
    balanced={t:float(np.mean([p.get(t,0.0) for p in comm_profiles.values()])) for t in vocab}
    balanced=normalize_counter(Counter(balanced))
    pooled_profile=normalize_counter(pooled)
    whole_profile=normalize_counter(whole)
    comm_summary=pd.DataFrame([{'year':year,'community':c,'authors':sum(1 for x in part.values() if x==c),
                                       'edge_support_papers':comm_papers[c],
                                       'token_weight':sum(comm_counts[c].values())}
                                      for c in sorted(comm_profiles)])
    return {
        'sub':sub,'graph':G,'lcc':L,'partition':part,'community_counts':comm_counts,
        'community_profiles':comm_profiles,'balanced_profile':balanced,
        'pooled_lcc_profile':pooled_profile,'pooled_lcc_counts':pooled,'whole_field_profile':whole_profile,'whole_field_counts':whole,
        'community_summary':comm_summary,'paper_assignments':pd.DataFrame(paper_rows),
        'edge_support_papers':edge_support_papers,
    }


def top_terms(profile,n=20):
    return sorted(profile.items(),key=lambda x:(-x[1],x[0]))[:n]


def profile_table(profile,year,profile_type,top_n=None):
    items=top_terms(profile,top_n) if top_n else sorted(profile.items(),key=lambda x:(-x[1],x[0]))
    return pd.DataFrame([{'year':year,'profile_type':profile_type,'rank':i+1,'term':t,'weight':w} for i,(t,w) in enumerate(items)])


def community_terms_table(result,top_n=25):
    rows=[]
    for c,p in result['community_profiles'].items():
        for rank,(t,w) in enumerate(top_terms(p,top_n),1):
            rows.append({'community':c,'rank':rank,'term':t,'normalized_weight':w,
                         'authors':sum(1 for x in result['partition'].values() if x==c)})
    return pd.DataFrame(rows)


def community_tfidf_table(result,top_n=25):
    counts=result['community_counts']; communities=sorted(counts); C=len(communities)
    df=Counter()
    for c in communities:
        for t,v in counts[c].items():
            if v>0: df[t]+=1
    rows=[]
    for c in communities:
        total=sum(counts[c].values())
        vals=[]
        for t,v in counts[c].items():
            tf=v/total if total else 0.0
            idf=math.log((1+C)/(1+df[t]))+1
            vals.append((t,tf*idf,tf,idf))
        vals.sort(key=lambda x:(-x[1],x[0]))
        chosen=vals if top_n is None else vals[:top_n]
        for rank,(t,score,tf,idf) in enumerate(chosen,1):
            rows.append({'community':c,'rank':rank,'term':t,'tfidf':score,'tf':tf,'idf':idf})
    return pd.DataFrame(rows)


def gini_positive(values):
    x=np.asarray([float(v) for v in values if float(v)>=0],dtype=float)
    if len(x)==0 or np.allclose(x.sum(),0): return np.nan
    x=np.sort(x); n=len(x)
    return float((2*np.sum((np.arange(1,n+1))*x)/(n*x.sum()))-(n+1)/n)


def vocabulary_concentration(profile,top_n=20):
    full=list(profile.values()); top=[v for _,v in top_terms(profile,top_n)]
    # Renormalize the displayed top-N distribution for a directly reproducible Gini.
    st=sum(top); top_norm=[v/st for v in top] if st else []
    return {'vocabulary_size':len(full),'gini_full_positive_vocabulary':gini_positive(full),
            f'gini_top{top_n}_renormalized':gini_positive(top_norm),
            f'top{top_n}_mass':float(st)}


def compare_top_lists(profile_a,profile_b,top_n=20):
    a=[t for t,_ in top_terms(profile_a,top_n)]; b=[t for t,_ in top_terms(profile_b,top_n)]
    A,B=set(a),set(b); inter=A&B; union=A|B
    return {'overlap_count':len(inter),'top_n_overlap_rate':len(inter)/top_n,
            'jaccard':len(inter)/len(union) if union else np.nan}


def rank_correlations(profile_a,profile_b,top_n=20):
    a=[t for t,_ in top_terms(profile_a,top_n)];b=[t for t,_ in top_terms(profile_b,top_n)]
    ra={t:i+1 for i,t in enumerate(a)};rb={t:i+1 for i,t in enumerate(b)}
    shared=sorted(set(a)&set(b))
    if len(shared)>=2:
        sp=spearmanr([ra[t] for t in shared],[rb[t] for t in shared]).statistic
        kt=kendalltau([ra[t] for t in shared],[rb[t] for t in shared]).statistic
    else: sp=kt=np.nan
    union=sorted(set(a)|set(b)); absent=len(union)+1
    if len(union)>=2:
        spu=spearmanr([ra.get(t,absent) for t in union],[rb.get(t,absent) for t in union]).statistic
        ktu=kendalltau([ra.get(t,absent) for t in union],[rb.get(t,absent) for t in union]).statistic
    else: spu=ktu=np.nan
    return {'shared_terms':len(shared),'spearman_shared_only':sp,'kendall_shared_only':kt,
            'union_terms':len(union),'absent_rank_rule':absent,
            'spearman_union_absent_last':spu,'kendall_union_absent_last':ktu}


def js_divergence(profile_a,profile_b):
    vocab=sorted(set(profile_a)|set(profile_b))
    if not vocab:return np.nan
    a=np.array([profile_a.get(t,0) for t in vocab],float);b=np.array([profile_b.get(t,0) for t in vocab],float)
    if a.sum()==0 or b.sum()==0:return np.nan
    # scipy returns distance = sqrt(JS divergence)
    return float(jensenshannon(a/a.sum(),b/b.sum())**2)


def aggregate_profiles(profiles):
    """Return equal-snapshot mean and pooled-snapshot (mass-weighted) alternatives."""
    years=sorted(profiles);vocab=set().union(*(profiles[y] for y in years))
    equal={t:float(np.mean([profiles[y].get(t,0.0) for y in years])) for t in vocab}
    equal=normalize_counter(Counter(equal))
    # Each profile is already normalized, so simple summation is mathematically the
    # same as equal-snapshot averaging up to a constant. A truly pooled profile must
    # be computed from paper/token counts; callers should supply it separately.
    return equal


def lcc_whole_field_comparison(result,top_n=20, lcc_profile='paper_pooled'):
    """Compare like-weighted LCC and whole-field title profiles.

    Representativeness is evaluated with paper-pooled profiles on both sides so
    the comparison is not confounded by the equal-community weighting used for
    the manuscript's principal LCC prevalence profile.  The legacy
    ``community_balanced`` option is retained only for explicit sensitivity
    work.
    """
    if lcc_profile == 'paper_pooled':
        a=result['pooled_lcc_profile']
    elif lcc_profile == 'community_balanced':
        a=result['balanced_profile']
    else:
        raise ValueError(f'Unknown LCC profile: {lcc_profile}')
    b=result['whole_field_profile']
    row=compare_top_lists(a,b,top_n);row.update(rank_correlations(a,b,top_n));row['js_divergence_full_profiles']=js_divergence(a,b)
    row['lcc_profile_type']=lcc_profile
    row['whole_field_profile_type']='paper_pooled'
    return row


def _wilson(k,n):
    if n==0:return (np.nan,np.nan)
    return proportion_confint(k,n,alpha=.05,method='wilson')


def snapshot_author_sets(df,snapshots=SNAPSHOTS,authors_col='authors_network',lcc_only=True):
    sets={}
    for y in snapshots:
        sub=filter_publications(df,y,y)
        G=build_coauthorship_graph(sub,authors_col=authors_col)
        H=largest_component_graph(G) if lcc_only else G
        sets[y]=set(H.nodes())
    return sets


def turnover_table(author_sets):
    years=sorted(author_sets);rows=[]
    for a,b in zip(years,years[1:]):
        A,B=author_sets[a],author_sets[b]; inter=A&B; gap=b-a
        surv=len(inter)/len(A) if A else np.nan; ret=len(inter)/len(B) if B else np.nan
        sl,sh=_wilson(len(inter),len(A));rl,rh=_wilson(len(inter),len(B))
        rows.append({'earlier_year':a,'later_year':b,'years_gap':gap,'earlier_authors':len(A),'later_authors':len(B),
                     'shared_authors':len(inter),'survival_rate':surv,'survival_ci_low':sl,'survival_ci_high':sh,
                     'retention_rate':ret,'retention_ci_low':rl,'retention_ci_high':rh,
                     'newcomer_share_later':1-ret if np.isfinite(ret) else np.nan,
                     'annualized_survival_equivalent':surv**(1/gap) if surv>0 else 0.0})
    return pd.DataFrame(rows)


def long_term_author_position(df,snapshots=SNAPSHOTS):
    sets=snapshot_author_sets(df,snapshots,'authors_network',True)
    common=set.intersection(*(sets[y] for y in snapshots))
    rows=[]
    for y in snapshots:
        sub=filter_publications(df,y,y);G=build_coauthorship_graph(sub);L=largest_component_graph(G)
        degrees=dict(L.degree()); frac=dict(L.degree(weight='fractional_weight'))
        degree_values=np.array(list(degrees.values()))
        for a in sorted(common):
            d=degrees.get(a,np.nan); fd=frac.get(a,np.nan)
            percentile=float((degree_values<=d).mean()) if np.isfinite(d) and len(degree_values) else np.nan
            rows.append({'year':y,'author':a,'degree':d,'fractional_degree':fd,'degree_percentile_in_lcc':percentile})
    cols=['year','author','degree','fractional_degree','degree_percentile_in_lcc']
    return common,pd.DataFrame(rows,columns=cols)


def _community_profiles_for_fixed_partition(sub, lcc, partition, exclusions, merge_map, extra_exclusions=None, remove_ordinals=True):
    """Compute the community-balanced title profile for an already fixed LCC partition.

    This helper is used for Louvain-seed sensitivity without rebuilding the
    co-authorship graph. Counting rules are identical to
    :func:`community_titleword_profiles`.
    """
    lcc_nodes=set(lcc.nodes())
    comm_counts=defaultdict(Counter)
    for _,r in sub.iterrows():
        toks=tokenize_title(r.get('title',''),exclusions,merge_map,extra_exclusions,remove_ordinals=remove_ordinals)
        authors=list(dict.fromkeys(a for a in r['authors_network'] if a in lcc_nodes))
        if len(authors)<2:
            continue
        communities=sorted(set(partition[a] for a in authors if a in partition))
        if not communities:
            continue
        frac=1.0/len(communities)
        for c in communities:
            for t in toks:
                comm_counts[c][t]+=frac
    comm_profiles={c:normalize_counter(cnt) for c,cnt in comm_counts.items()}
    vocab=set().union(*(p.keys() for p in comm_profiles.values())) if comm_profiles else set()
    balanced={t:float(np.mean([p.get(t,0.0) for p in comm_profiles.values()])) for t in vocab}
    return normalize_counter(Counter(balanced))


def louvain_seed_sensitivity(df, year, exclusions, merge_map, seeds=(0,1,2,3,4,5,10,20,30,42), reference_seed=42, top_n=20):
    """Assess sensitivity of the community-balanced title profile to Louvain randomness.

    The co-authorship graph and LCC are held fixed. For each seed, Louvain is
    rerun with the same weighted graph and the resulting community-balanced
    titleword profile is compared to the reference-seed profile. This is a
    robustness diagnostic, not a procedure for choosing the most favourable
    partition.
    """
    sub,G,L,_=snapshot_social_structure(df,year,seed=reference_seed)
    ref_part=louvain_partition(L,seed=reference_seed,weight='paper_count')
    ref_profile=_community_profiles_for_fixed_partition(sub,L,ref_part,exclusions,merge_map)
    rows=[]
    for seed in seeds:
        part=louvain_partition(L,seed=int(seed),weight='paper_count')
        profile=_community_profiles_for_fixed_partition(sub,L,part,exclusions,merge_map)
        communities=[]
        for cid in sorted(set(part.values())):
            communities.append({n for n,c in part.items() if c==cid})
        modularity=nx.community.modularity(L,communities,weight='paper_count') if L.number_of_edges() else np.nan
        cmp=compare_top_lists(ref_profile,profile,top_n)
        rows.append({
            'year':int(year),'seed':int(seed),'reference_seed':int(reference_seed),
            'n_communities':len(communities),'weighted_modularity':float(modularity),
            'top20_overlap_with_reference':cmp['top_n_overlap_rate'],
            'top20_jaccard_with_reference':cmp['jaccard'],
            'js_divergence_from_reference':js_divergence(ref_profile,profile),
        })
    return pd.DataFrame(rows)
