from pathlib import Path
import tempfile

import pandas as pd
import networkx as nx

from fusion_pipeline.wos_parser import parse_wos_file
from fusion_pipeline.normalization import (
    normalize_country_from_affiliation,
    is_jet_affiliation,
    build_affiliation_table,
)
from fusion_pipeline.disambiguation import build_author_profiles, score_candidate
from fusion_pipeline.networks import build_coauthorship_graph
from fusion_pipeline.internationalization import prepare_country_dataframe, yearly_icr
from fusion_pipeline.socio_epistemic import compare_top_lists
from fusion_pipeline.graph_export import (
    GraphExportOptions,
    enrich_projection_graph,
    write_graphml_bundle,
)


def test_wos_parser_preserves_author_boundaries_and_commas():
    text = """PT J\nAU Smith, AB\n   Jones, CD\n   Van der Meer, E\nAF Smith, Alice B.\n   Jones, Charles D.\n   Van der Meer, Eva\nTI Test paper\nSO Test Journal\nDT Article\nPY 1986\nUT WOS:TEST001\nER\n"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample.txt"
        p.write_text(text, encoding="utf-8")
        rec = parse_wos_file(p)[0]
    assert rec["authors_wos"] == ["Smith, AB", "Jones, CD", "Van der Meer, E"]
    assert rec["authors"] == ["SMITH, AB", "JONES, CD", "VAN DER MEER, E"]
    assert rec["n_authors"] == 3


def test_wos_parser_preserves_reprint_address_for_missingness_audit():
    text = """PT J\nAU Smith, AB\nTI Test paper\nSO Test Journal\nDT Article\nPY 1986\nRP Smith, AB (reprint author), PPPL, Princeton, NJ 08544 USA.\nUT WOS:TEST-RP\nER\n"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sample.txt"
        p.write_text(text, encoding="utf-8")
        rec = parse_wos_file(p)[0]
    assert "PPPL" in rec["reprint_address_one_line"]


def test_country_normalization_is_field_local_and_uk_aliases_are_normalized():
    assert normalize_country_from_affiliation("Culham Lab, Abingdon, England.")[0] == "UK"
    assert normalize_country_from_affiliation("Institute X, Glasgow, Scotland.")[0] == "UK"
    assert normalize_country_from_affiliation("Institute X, Cardiff, Wales.")[0] == "UK"
    assert normalize_country_from_affiliation("Institute X, London, United Kingdom.")[0] == "UK"
    # A country-like substring embedded in another word must not create USA.
    assert normalize_country_from_affiliation("Natl Inst Fusion Sci, Chikusa, Nagoya, Japan.")[0] == "JAPAN"


def test_usa_state_heuristic_can_be_disabled_for_parser_sensitivity():
    address="Princeton Plasma Physics Laboratory, Princeton, NJ 08544."
    assert normalize_country_from_affiliation(address)[0] == "USA"
    assert normalize_country_from_affiliation(address,allow_us_state_heuristic=False)[0] is None


def test_jet_network_node_only_for_joint_european_torus_affiliations():
    assert is_jet_affiliation("JET Joint Undertaking, Abingdon, England.")
    assert is_jet_affiliation("EFDA JET Close Support Unit, Culham, England.")
    assert not is_jet_affiliation("Jet Propulsion Laboratory, Pasadena, CA 91109, USA.")
    assert not is_jet_affiliation("Aerojet, Sacramento, CA, USA.")

    records = [{
        "record_id": "WOS:J1",
        "year": 1986,
        "affiliations": [
            "JET Joint Undertaking, Abingdon, England.",
            "Culham Laboratory, Abingdon, England.",
            "Princeton Plasma Physics Laboratory, Princeton, NJ 08544, USA.",
        ],
    }]
    aff = build_affiliation_table(records)
    assert aff["country_normalized"].tolist() == ["UK", "UK", "USA"]
    assert aff["network_country"].tolist() == ["JET", "UK", "USA"]


def test_disambiguation_blocks_same_paper_and_large_team_only_context():
    # Same-paper co-occurrence is direct evidence that two name forms are distinct.
    same_paper = [{
        "record_id": "R1", "year": 1986,
        "authors_normalized": ["SMITH, A", "SMITH, AB", "X, C"],
        "institutions_normalized": ["INST"], "authors_full_wos": [],
    }]
    prof = build_author_profiles(same_paper)
    scored = score_candidate(prof["SMITH, A"], prof["SMITH, AB"])
    assert scored["auto_merge"] is False
    assert scored["auto_rule"] == "BLOCKED_cooccurrence_same_paper"

    # A shared coauthor that occurs only in a >20-author paper is not decisive.
    team = [f"TEAM{i}, A" for i in range(21)]
    records = [
        {"record_id": "R2", "year": 1986, "authors_normalized": ["ROSS, D"] + team,
         "institutions_normalized": ["UT AUSTIN"], "authors_full_wos": []},
        {"record_id": "R3", "year": 1987, "authors_normalized": ["ROSS, DW"] + team,
         "institutions_normalized": ["UT AUSTIN"], "authors_full_wos": []},
    ]
    prof = build_author_profiles(records)
    scored = score_candidate(prof["ROSS, D"], prof["ROSS, DW"])
    assert scored["shared_coauthors"] >= 1
    assert scored["shared_coauthors_small_team"] == 0
    assert scored["auto_merge"] is False


def test_coauthorship_fractional_weight_is_newman_style():
    df = pd.DataFrame([{"record_id": "P1", "authors_network": ["A", "B", "C"]}])
    g = build_coauthorship_graph(df)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert g["A"]["B"]["paper_count"] == 1
    assert g["A"]["B"]["fractional_weight"] == 0.5


def test_rich_graph_export_retains_exact_edge_publication_provenance(tmp_path):
    df = pd.DataFrame([
        {"record_id": "P1", "year": 1968, "title": "Stellarator plasma confinement",
         "authors_network": ["A", "B", "C"], "institutions_normalized": ["LAB 1"],
         "countries_normalized": ["UK"], "keywords": ["stellarator"],
         "document_types": ["Article"]},
        {"record_id": "P2", "year": 1968, "title": "Tokamak plasma heating",
         "authors_network": ["A", "B"], "institutions_normalized": ["LAB 2"],
         "countries_normalized": ["USA"], "keywords": ["tokamak"],
         "document_types": ["Article"]},
        {"record_id": "P3", "year": 1968, "title": "Unrelated solo paper",
         "authors_network": ["A"], "institutions_normalized": ["LAB 3"],
         "countries_normalized": ["FRANCE"], "keywords": [],
         "document_types": ["Article"]},
    ])
    graph = build_coauthorship_graph(df)
    rich, audit = enrich_projection_graph(
        graph, df, options=GraphExportOptions(profile="exploration"),
        title_exclusions={"plasma"}, graph_kind="author", start_year=1968,
        end_year=1968,
    )
    assert rich.is_directed() is False
    assert rich["A"]["B"]["record_ids"] == ["P1", "P2"]
    assert rich["A"]["B"]["titles"] == [
        "Stellarator plasma confinement", "Tokamak plasma heating",
    ]
    assert "Unrelated solo paper" not in rich["A"]["B"]["titles"]
    assert rich["A"]["C"]["record_ids"] == ["P1"]
    assert "stellarator" in rich["A"]["C"]["top_terms"]
    assert audit["unsupported_edges"] == []
    assert audit["paper_count_mismatches"] == []

    paths = write_graphml_bundle(rich, tmp_path / "rich", ("graphml", "graphmlz"))
    assert {path.suffix for path in paths} == {".graphml", ".graphmlz"}
    round_trip = nx.read_graphml(tmp_path / "rich.graphml")
    assert round_trip.is_directed() is False
    assert "titles" in round_trip["A"]["B"]


def test_custom_graph_export_profile_is_user_selectable():
    options = GraphExportOptions(
        profile="custom", node_attributes=("label", "topic_label"),
        edge_attributes=("titles", "years"),
    )
    nodes, edges = options.selected_attributes()
    assert nodes == {"label", "topic_label"}
    assert edges == {"titles", "years"}


def test_historical_record_exclusion_is_exact_period_scoped(tmp_path):
    from fusion_pipeline.networks import apply_exact_period_record_exclusions
    rules=tmp_path/'exclusions.csv'
    pd.DataFrame([{'analysis':'coauthor_snapshot','start_year':1969,'end_year':1973,
                   'record_id':'X','scope':'exact_period_only','reason':'audit','checked_by':'author'}]).to_csv(rules,index=False)
    df=pd.DataFrame([{'record_id':'X','year':1971},{'record_id':'Y','year':1971}])
    exact,used=apply_exact_period_record_exclusions(df,str(rules),1969,1973)
    shifted,unused=apply_exact_period_record_exclusions(df,str(rules),1970,1974)
    assert exact.record_id.tolist()==['Y'] and len(used)==1
    assert shifted.record_id.tolist()==['X','Y'] and unused.empty


def test_icr_uses_geographic_countries_not_jet_network_node():
    df = pd.DataFrame([
        {"year": 1986, "countries_normalized": ["UK"], "countries_network": ["JET"],
         "authors_network": ["A"], "institutions": [], "document_types": ["Article"], "affiliations": ["JET"]},
        {"year": 1986, "countries_normalized": ["UK", "USA"], "countries_network": ["JET", "USA"],
         "authors_network": ["B", "C"], "institutions": [], "document_types": ["Article"], "affiliations": ["JET", "PPPL"]},
    ])
    prep = prepare_country_dataframe(df)
    icr = yearly_icr(prep).iloc[0]
    # JET-only is geographically domestic UK; the UK-USA paper is international.
    assert icr["papers_with_country"] == 2
    assert icr["international_papers"] == 1
    assert icr["ICR"] == 0.5


def test_top20_overlap_is_symmetric_for_fixed_length_lists():
    a = {f"a{i}": 100 - i for i in range(20)}
    b = {**{f"a{i}": 100 - i for i in range(8)}, **{f"b{i}": 50 - i for i in range(12)}}
    ab = compare_top_lists(a, b, top_n=20)
    ba = compare_top_lists(b, a, top_n=20)
    assert ab["overlap_count"] == 8
    assert ab["top_n_overlap_rate"] == ba["top_n_overlap_rate"]
    assert ab["jaccard"] == ba["jaccard"]

from fusion_pipeline.exploration import make_time_windows, longitudinal_author_metrics, compute_centralities, compute_graph_metrics


def test_user_selected_time_windows_respect_exact_range():
    annual = make_time_windows(1974, 1978, mode="annual")
    assert annual[0].start_year == 1974 and annual[-1].end_year == 1978
    assert all(w.start_year >= 1974 and w.end_year <= 1978 for w in annual)
    rolling = make_time_windows(1974, 1978, mode="rolling", window_size=3)
    assert [(w.start_year, w.end_year) for w in rolling] == [(1974,1976),(1975,1977),(1976,1978)]
    period = make_time_windows(1974, 1978, mode="period")
    assert [(w.start_year, w.end_year) for w in period] == [(1974,1978)]


def test_longitudinal_metrics_do_not_leak_outside_selected_years():
    df = pd.DataFrame([
        {"year":1973,"authors_network":["OLD","A"],"document_types":["Article"],"team_size":2},
        {"year":1974,"authors_network":["A","B"],"document_types":["Article"],"team_size":2},
        {"year":1975,"authors_network":["B","C"],"document_types":["Article"],"team_size":2},
        {"year":1976,"authors_network":["C","D"],"document_types":["Article"],"team_size":2},
        {"year":1977,"authors_network":["D","E"],"document_types":["Article"],"team_size":2},
        {"year":1978,"authors_network":["E","F"],"document_types":["Article"],"team_size":2},
        {"year":1979,"authors_network":["F","NEW"],"document_types":["Article"],"team_size":2},
    ])
    out = longitudinal_author_metrics(df,1974,1978,mode="period",metrics=["number_of_nodes","number_of_edges"])
    assert out.iloc[0]["number_of_nodes"] == 6
    assert out.iloc[0]["number_of_edges"] == 5


def test_extended_metrics_and_legacy_centralities_are_available():
    g = nx.Graph()
    g.add_edge("A","B",weight=1,fractional_weight=.5)
    g.add_edge("B","C",weight=1,fractional_weight=.5)
    metrics = compute_graph_metrics(g,["density","diameter_of_largest_component","average_shortest_path_length_lcc","degree_assortativity","modularity"])
    assert metrics["density"] > 0
    assert metrics["diameter_of_largest_component"] == 2
    cent = compute_centralities(g,["degree","degree_normalized","weighted_degree","fractional_degree","betweenness","closeness","eigenvector","eccentricity"],scope="full")
    assert set(["degree","closeness","eigenvector","eccentricity"]).issubset(cent.columns)
    assert cent.loc[cent.node=="B","betweenness"].iloc[0] > cent.loc[cent.node=="A","betweenness"].iloc[0]


def test_historical_country_output_keeps_dynamic_euratom_and_combined_ussr_russia():
    from fusion_pipeline.historical_figures import aggregate_fractional_country_output
    df = pd.DataFrame([
        {"year":1974,"countries_normalized":["FRANCE"],"countries_network":["FRANCE"],"authors_network":["A"],"institutions":[],"document_types":["Article"],"affiliations":["x"]},
        {"year":1974,"countries_normalized":["SPAIN"],"countries_network":["SPAIN"],"authors_network":["B"],"institutions":[],"document_types":["Article"],"affiliations":["x"]},
        {"year":1974,"countries_normalized":["USSR"],"countries_network":["USSR"],"authors_network":["C"],"institutions":[],"document_types":["Article"],"affiliations":["x"]},
        {"year":1993,"countries_normalized":["RUSSIA"],"countries_network":["RUSSIA"],"authors_network":["D"],"institutions":[],"document_types":["Article"],"affiliations":["x"]},
    ])
    out=aggregate_fractional_country_output(df,1974,1993,["EURATOM","USSR_RUSSIA"])
    eur1974=out[(out.year==1974)&(out.series=="EURATOM")].fractional_papers.iloc[0]
    sov1974=out[(out.year==1974)&(out.series=="USSR_RUSSIA")].fractional_papers.iloc[0]
    sov1993=out[(out.year==1993)&(out.series=="USSR_RUSSIA")].fractional_papers.iloc[0]
    assert eur1974 == 1.0  # France counts; Spain joined later.
    assert sov1974 == 1.0 and sov1993 == 1.0


def test_fixed_europe_output_separates_growth_from_enlargement():
    from fusion_pipeline.historical_figures import aggregate_fractional_country_output
    df = pd.DataFrame([
        {"year":1974,"countries_normalized":["FRANCE"],"countries_network":["FRANCE"],"authors_network":["A"],"institutions":[],"document_types":["Article"],"affiliations":["x"]},
        {"year":1974,"countries_normalized":["SPAIN"],"countries_network":["SPAIN"],"authors_network":["B"],"institutions":[],"document_types":["Article"],"affiliations":["x"]},
    ])
    dyn=aggregate_fractional_country_output(df,1974,1974,["EURATOM"],"dynamic","EC12_1986")
    fixed=aggregate_fractional_country_output(df,1974,1974,["EURATOM"],"fixed","EC12_1986")
    assert dyn.fractional_papers.iloc[0] == 1.0
    assert fixed.fractional_papers.iloc[0] == 2.0


def test_focal_partner_robustness_can_exclude_jet():
    from fusion_pipeline.internationalization import prepare_country_dataframe, focal_country_intra_extra
    df=pd.DataFrame([{
        "year":1988,"countries_normalized":["UK"],"countries_network":["UK","JET"],
        "authors_network":["A","B"],"institutions":[],"document_types":["Article"],"affiliations":["x"],
    }])
    prep=prepare_country_dataframe(df)
    _,with_jet=focal_country_intra_extra(prep,"UK",1988,1988,include_jet_partner=True)
    _,without_jet=focal_country_intra_extra(prep,"UK",1988,1988,include_jet_partner=False)
    assert with_jet.papers_with_euratom_partner.iloc[0] == 1
    assert without_jet.papers_with_euratom_partner.iloc[0] == 0


def test_period_contrast_uses_user_selected_nonoverlapping_periods():
    from fusion_pipeline.period_contrast import prepare_period_contrast, add_feature_column, distinctive_features
    df=pd.DataFrame([
        {'year':1970,'title':'stellarator confinement','authors_network':['A'],'institutions_normalized':['X'],'countries_normalized':['USA']},
        {'year':1971,'title':'stellarator plasma','authors_network':['B'],'institutions_normalized':['X'],'countries_normalized':['USA']},
        {'year':1974,'title':'tokamak heating','authors_network':['C'],'institutions_normalized':['Y'],'countries_normalized':['FRANCE']},
        {'year':1975,'title':'tokamak reactor','authors_network':['D'],'institutions_normalized':['Y'],'countries_normalized':['FRANCE']},
        {'year':1979,'title':'outside range','authors_network':['E'],'institutions_normalized':['Z'],'countries_normalized':['UK']},
    ])
    sub=prepare_period_contrast(df,1974,1975,1970,1971)
    assert set(sub.year)=={1970,1971,1974,1975}
    sub,col=add_feature_column(sub,'title',(1,))
    dist=distinctive_features(sub,col,min_doc_freq=1)
    tok=dist[dist.feature.eq('tokamak')].iloc[0]
    assert tok.focal_count==2 and tok.baseline_count==0


def test_publication_trend_comparison_accepts_arbitrary_baselines():
    from fusion_pipeline.publication_trends import compare_publication_series, detect_common_trend_periods
    years=list(range(2000,2006))
    fusion=pd.DataFrame({'year':years,'count':[10,12,15,19,24,30]})
    physics=pd.DataFrame({'year':years,'count':[100,100,100,100,100,100]})
    engineering=pd.DataFrame({'year':years,'count':[200,200,200,200,200,200]})
    comp=compare_publication_series({'fusion':fusion,'physics':physics,'engineering':engineering},smooth_window=1)
    assert {'fusion_vs_physics','fusion_vs_engineering'}.issubset(comp.columns)
    periods=detect_common_trend_periods(comp,['fusion_vs_physics','fusion_vs_engineering'],min_consecutive_years=3,min_avg_slope=0)
    assert not periods.empty and (periods.direction=='increase').all()


def test_manual_historical_annotations_match_exact_period_only(tmp_path):
    from fusion_pipeline.historical_annotations import load_period_annotations
    p = tmp_path / "annotations.csv"
    pd.DataFrame([
        {"network":"coauthor_lcc","start_year":1974,"end_year":1978,"label":"LAB A","anchor_nodes":"A;B","label_dx_points":10,"label_dy_points":20,"provenance":"manual"},
        {"network":"coauthor_lcc","start_year":1986,"end_year":1986,"label":"LAB B","anchor_nodes":"C","label_dx_points":10,"label_dy_points":20,"provenance":"manual"},
    ]).to_csv(p,index=False)
    exact = load_period_annotations(p,1974,1978)
    shifted = load_period_annotations(p,1975,1979)
    assert exact.label.tolist() == ["LAB A"]
    assert shifted.empty


def test_manual_annotation_anchor_audit_does_not_infer_missing_labs():
    from fusion_pipeline.historical_annotations import validate_annotation_anchors
    g=nx.Graph(); g.add_edge("A","B")
    ann=pd.DataFrame([
        {"label":"KNOWN","anchor_nodes":"A;Z"},
        {"label":"UNKNOWN","anchor_nodes":"X;Y"},
    ])
    audit=validate_annotation_anchors(g,ann)
    known=audit[audit.label.eq("KNOWN")].iloc[0]
    unknown=audit[audit.label.eq("UNKNOWN")].iloc[0]
    assert known.status == "ok" and known.matched_anchors == "A" and known.missing_anchors == "Z"
    assert unknown.status == "no_anchor_in_graph" and unknown.matched_anchors == ""


def test_publication_trend_counts_can_be_derived_from_record_corpus():
    from fusion_pipeline.publication_trends import annual_publication_counts
    df = pd.DataFrame({'year':[2000,2000,2001,2002,2002,2002], 'x':range(6)})
    out = annual_publication_counts(df, start_year=2000, end_year=2002)
    assert out.to_dict('records') == [
        {'year':2000,'count':2}, {'year':2001,'count':1}, {'year':2002,'count':3}
    ]


def test_publication_trend_excluded_year_breaks_adjacent_run():
    from fusion_pipeline.publication_trends import compare_publication_series, detect_common_trend_periods
    years=list(range(2000,2007))
    fusion=pd.DataFrame({'year':years,'count':[10,20,40,80,160,320,640]})
    p=pd.DataFrame({'year':years,'count':[100]*7})
    comp=compare_publication_series({'fusion':fusion,'physics':p,'engineering':p},smooth_window=1)
    periods=detect_common_trend_periods(comp,['fusion_vs_physics','fusion_vs_engineering'],min_consecutive_years=2,min_avg_slope=0,exclude_years=[2003])
    # The excluded value and its following first difference must split the monotonic run.
    assert not ((periods.start_year <= 2003) & (periods.end_year >= 2004)).any()


def test_guzzardi_multipanel_stats_and_render_are_descriptive_only(tmp_path, monkeypatch):
    from fusion_pipeline import guzzardi_figures as gf

    # Small deterministic graphs: avoid dependence on Graphviz in this unit test.
    graphs={}
    for year,n in zip([1979,1986,1996,2001],[6,8,10,12]):
        g=nx.Graph()
        g.add_nodes_from(range(n))
        # LCC of n-2 nodes; final two nodes form their own dyad.
        g.add_edges_from((i,i+1) for i in range(n-3))
        g.add_edge(n-2,n-1)
        graphs[year]=g

    monkeypatch.setattr(gf,'sfdp_layout',lambda G,seed=42,pack_components=False: nx.spring_layout(G,seed=seed))
    out=tmp_path/'multipanel.png'
    audit=gf.plot_full_network_multipanel(graphs,out,years=[1979,1986,1996,2001],dpi=80)
    assert out.exists() and out.stat().st_size > 0
    assert audit.year.tolist() == [1979,1986,1996,2001]
    assert audit.components.tolist() == [2,2,2,2]
    assert audit.lcc_nodes.tolist() == [4,6,8,10]
    # Multipanel audit contains no Louvain or betweenness variables by design.
    assert 'communities' not in audit.columns
    assert not any('betweenness' in c for c in audit.columns)


def test_community_vocabulary_selection_and_plot(tmp_path):
    from collections import Counter
    from fusion_pipeline.community_vocabulary import community_vocabulary_figure_data, plot_community_vocabulary_panels
    result={
        'community_summary': pd.DataFrame([
            {'year':1979,'community':0,'authors':40,'edge_support_papers':20},
            {'year':1979,'community':1,'authors':30,'edge_support_papers':15},
            {'year':1979,'community':2,'authors':10,'edge_support_papers':8},
        ]),
        'community_counts': {
            0: Counter({'tokamak':10,'isx-b':8,'plasma':4}),
            1: Counter({'tokamak':9,'plt':8,'plasma':5}),
            2: Counter({'mirror':7,'plasma':3}),
        }
    }
    d=community_vocabulary_figure_data(result,1979,top_communities=2,terms_per_community=2)
    assert set(d.community)=={0,1}
    assert 'isx-b' in set(d.term) and 'plt' in set(d.term)
    panels={y:d.assign(year=y) for y in [1979,1986,1996,2001]}
    out=tmp_path/'f5.png'
    plot_community_vocabulary_panels(panels,out,dpi=80)
    assert out.exists() and out.stat().st_size>0


def test_submission_titleword_preprocessing_is_explicit_and_markup_safe(tmp_path):
    from fusion_pipeline.socio_epistemic import tokenize_title
    stop={'the','and','fusion'}
    merge={'plasmas':'plasma'}
    title='The FUSION response of Li<sub>2</sub>O plasmas at 2nd harmonic and electron-cyclotron heating'
    toks=tokenize_title(title,stop,merge,remove_ordinals=True)
    assert 'fusion' not in toks
    assert 'sub' not in toks and 'sup' not in toks
    assert '2nd' not in toks
    assert 'li2o' in toks
    assert 'plasma' in toks
    assert 'electron-cyclotron' in toks
    assert 'harmonic' in toks


def test_ordinal_preservation_is_a_separate_sensitivity_mode():
    from fusion_pipeline.socio_epistemic import tokenize_title
    toks=tokenize_title('2nd harmonic 1st-wall',set(),{},remove_ordinals=False)
    assert '2nd' in toks and '1st-wall' in toks
    cleaned=tokenize_title('2nd harmonic 1st-wall',set(),{},remove_ordinals=True)
    assert '2nd' not in cleaned and 'wall' in cleaned and 'harmonic' in cleaned


def test_louvain_partition_is_insertion_order_deterministic():
    from fusion_pipeline.networks import louvain_partition
    edges=[('a','b'),('b','c'),('c','a'),('d','e'),('e','f'),('f','d'),('c','d')]
    g1=nx.Graph(); g1.add_edges_from(edges)
    g2=nx.Graph(); g2.add_edges_from(reversed(edges))
    p1=louvain_partition(g1,seed=42)
    p2=louvain_partition(g2,seed=42)
    assert p1==p2


def test_exploration_community_table_is_insertion_order_deterministic():
    import networkx as nx
    from fusion_pipeline.exploration import community_table
    edges=[("A","B",2.0),("B","C",1.0),("C","A",1.0),("D","E",2.0),("E","F",1.0),("F","D",1.0),("C","D",0.2)]
    g1=nx.Graph(); g2=nx.Graph()
    for u,v,w in edges: g1.add_edge(u,v,weight=w)
    for u,v,w in reversed(edges): g2.add_edge(v,u,weight=w)
    t1=community_table(g1,scope="full",seed=42).sort_values("node").reset_index(drop=True)
    t2=community_table(g2,scope="full",seed=42).sort_values("node").reset_index(drop=True)
    assert t1[["node","community"]].equals(t2[["node","community"]])


def test_exploration_louvain_modularity_is_insertion_order_deterministic():
    import networkx as nx
    from fusion_pipeline.exploration import compute_graph_metrics
    edges=[("A","B",2.0),("B","C",1.0),("C","A",1.0),("D","E",2.0),("E","F",1.0),("F","D",1.0),("C","D",0.2)]
    g1=nx.Graph(); g2=nx.Graph()
    for u,v,w in edges: g1.add_edge(u,v,weight=w)
    for u,v,w in reversed(edges): g2.add_edge(v,u,weight=w)
    m1=compute_graph_metrics(g1,metrics=("modularity",),modularity_method="louvain",seed=42)
    m2=compute_graph_metrics(g2,metrics=("modularity",),modularity_method="louvain",seed=42)
    assert m1==m2


def test_historical_plot_accepts_frozen_layout(tmp_path):
    import networkx as nx
    import pandas as pd
    from fusion_pipeline.historical_annotations import plot_annotated_historical_lcc
    g=nx.Graph(); g.add_edge("A","B",weight=1.0)
    out=tmp_path/"frozen_layout.png"
    ann=pd.DataFrame(columns=["label","anchor_nodes","label_dx_points","label_dy_points"])
    plot_annotated_historical_lcc(g,{"A":0,"B":1},{"A":0.1,"B":0.2},ann,2000,2000,out,annotation_mode="none",layout_override={"A":(-1.0,0.0),"B":(1.0,0.0)},dpi=72)
    assert out.exists() and out.stat().st_size>0


def test_historical_soviet_republic_country_normalization_and_georgia_disambiguation():
    from fusion_pipeline.normalization import normalize_country_from_affiliation

    # Soviet-republic labels are represented as USSR through 1991.
    assert normalize_country_from_affiliation(
        "SUKHUMI PHYSICO TECH INST, SUKHUMI, GEORGIA.", year=1973
    )[0] == "USSR"
    assert normalize_country_from_affiliation(
        "ACAD SCI BESSR, INST PHYS, MINSK, BELARUS.", year=1979
    )[0] == "USSR"
    assert normalize_country_from_affiliation(
        "ACAD SCI GESSR, INST PHYS, TBILISI, GESSR.", year=1983
    )[0] == "USSR"

    # Successor-state labels begin in 1992 for year-granularity analysis.
    assert normalize_country_from_affiliation(
        "SUKHUMI INST PHYS & TECHNOL, SUKHUMI, GA.", year=1993
    )[0] == "GEORGIA"
    assert normalize_country_from_affiliation(
        "Sukhumi Physicotech Inst, GE-384914 Sukhumi, Abkhazia, Georgia.", year=2007
    )[0] == "GEORGIA"

    # Georgia in an institution name must not defeat the U.S. GA + ZIP heuristic.
    assert normalize_country_from_affiliation(
        "GEORGIA INST TECHNOL, NUCL ENGN PROGRAM, ATLANTA, GA 30332.", year=1986
    )[0] == "USA"


def test_all_soviet_republic_labels_and_successor_cutover():
    from fusion_pipeline.normalization import normalize_country_from_affiliation
    soviet_labels = [
        "RSFSR", "UKRAINIAN SSR", "BYELORUSSIAN SSR", "UZBEK SSR",
        "KAZAKH SSR", "GEORGIAN SSR", "AZERBAIJAN SSR", "LITHUANIAN SSR",
        "MOLDAVIAN SSR", "LATVIAN SSR", "KIRGHIZ SSR", "TAJIK SSR",
        "ARMENIAN SSR", "TURKMEN SSR", "ESTONIAN SSR",
    ]
    for label in soviet_labels:
        country, _ = normalize_country_from_affiliation(f"INST, CITY, {label}.", year=1988)
        assert country == "USSR"
    modern = {
        "RUSSIA":"RUSSIA", "UKRAINE":"UKRAINE", "BELARUS":"BELARUS",
        "UZBEKISTAN":"UZBEKISTAN", "KAZAKHSTAN":"KAZAKHSTAN",
        "GEORGIA":"GEORGIA", "AZERBAIJAN":"AZERBAIJAN", "LITHUANIA":"LITHUANIA",
        "MOLDOVA":"MOLDOVA", "LATVIA":"LATVIA", "KYRGYZSTAN":"KYRGYZSTAN",
        "TAJIKISTAN":"TAJIKISTAN", "ARMENIA":"ARMENIA",
        "TURKMENISTAN":"TURKMENISTAN", "ESTONIA":"ESTONIA",
    }
    for token, expected in modern.items():
        c1991, _ = normalize_country_from_affiliation(f"INST, CITY, {token}.", year=1991)
        c1992, _ = normalize_country_from_affiliation(f"INST, CITY, {token}.", year=1992)
        assert c1991 == "USSR"
        assert c1992 == expected


def test_manuscript_icr_uses_centered_pooled_ratio_without_change_point_fit():
    import pandas as pd
    from fusion_pipeline.historical_figures import icr_series_for_manuscript
    df = pd.DataFrame({
        "year":[2000,2000,2001,2001,2002,2002],
        "countries_normalized":[
            '["USA"]','["USA","FRANCE"]','["USA"]','["USA","GERMANY"]','["FRANCE"]','["FRANCE","GERMANY"]'
        ],
        "countries_network":[
            '["USA"]','["USA","FRANCE"]','["USA"]','["USA","GERMANY"]','["FRANCE"]','["FRANCE","GERMANY"]'
        ],
    })
    out=icr_series_for_manuscript(df,2000,2002,3)
    assert "ICR_centered_pooled" in out.columns
    assert out.loc[out.year.eq(2000), "ICR_centered_pooled"].isna().all()
    assert out.loc[out.year.eq(2001), "ICR_centered_pooled"].iloc[0] == 0.5
    assert out.loc[out.year.eq(2002), "ICR_centered_pooled"].isna().all()
    assert not any("fit" in c.lower() or "bic" in c.lower() for c in out.columns)


def test_post_1991_legacy_ussr_addresses_use_successor_state():
    from fusion_pipeline.normalization import normalize_country_from_affiliation
    cases = {
        "PN LEBEDEV PHYS INST, MOSCOW, USSR.": "RUSSIA",
        "ACAD SCI UKSSR, INST NUCL RES, KIEV, UKRAINE, USSR.": "UKRAINE",
        "V KAPSUKAS STATE UNIV, VILNIUS, LITHUANIA, USSR.": "LITHUANIA",
    }
    for address, expected in cases.items():
        country, _ = normalize_country_from_affiliation(address, year=1992)
        assert country == expected
    unresolved, method = normalize_country_from_affiliation("UNKNOWN INST, UNKNOWN CITY, USSR.", year=1992)
    assert unresolved is None
    assert method == "post_soviet_legacy_unresolved"


def test_guzzardi_titleword_author_proportional_allocation_differs_from_equal_communities():
    import networkx as nx
    import pandas as pd
    from fusion_pipeline.socio_epistemic import community_titleword_profiles
    # One paper has two authors in community 0 and one in community 1.
    df = pd.DataFrame([
        {"year":2000,"title":"alpha beta","authors_network":["A","B","C"],"record_id":"r1"},
        {"year":2000,"title":"gamma","authors_network":["A","B"],"record_id":"r2"},
    ])
    G=nx.Graph(); G.add_edges_from([("A","B"),("A","C"),("B","C")])
    social=(df,G,G,{"A":0,"B":0,"C":1})
    eq=community_titleword_profiles(df,2000,set(),{},social=social,allocation_mode='equal_communities')
    ap=community_titleword_profiles(df,2000,set(),{},social=social,allocation_mode='author_proportional')
    assert eq['community_counts'][0]['alpha'] == 0.5
    assert abs(ap['community_counts'][0]['alpha'] - 2/3) < 1e-12
    assert abs(ap['community_counts'][1]['alpha'] - 1/3) < 1e-12


def test_guzzardi_lcc_representativeness_uses_like_weighted_pooled_profiles():
    from fusion_pipeline.socio_epistemic import lcc_whole_field_comparison
    result={
        'balanced_profile':{'balanced_only':1.0},
        'pooled_lcc_profile':{'shared':0.8,'lcc_only':0.2},
        'whole_field_profile':{'shared':0.7,'whole_only':0.3},
    }
    out=lcc_whole_field_comparison(result,top_n=2)
    assert out['lcc_profile_type'] == 'paper_pooled'
    assert out['whole_field_profile_type'] == 'paper_pooled'
    assert out['overlap_count'] == 1
