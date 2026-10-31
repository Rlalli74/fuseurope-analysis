"""Rich, auditable GraphML/GraphMLZ exports for network exploration.

The manuscript analyses need comparatively small structural graphs, whereas
historical exploration benefits from retaining the publications behind every
node and edge.  This module keeps those purposes separate through named export
profiles and user-selectable attribute lists.  It never changes the graph used
for a calculation: it only adds provenance, descriptive text features, and
selected structural measures to a copy.

``top_terms`` and ``topic_label`` are deliberately descriptive title-term
summaries (frequency-ranked unigrams and bigrams), not a fitted topic model.
They are intended as navigation aids for close reading.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import gzip
import json
import math
import tempfile

import networkx as nx
import numpy as np
import pandas as pd

from .networks import largest_component_graph, louvain_partition, parse_list_cell
from .socio_epistemic import tokenize_title


PROFILE_ATTRIBUTES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "minimal": {
        "node": ("label", "paper_count", "degree"),
        "edge": ("paper_count", "weight", "fractional_weight"),
    },
    "manuscript": {
        "node": (
            "label", "paper_count", "degree", "weighted_degree",
            "fractional_degree", "betweenness_centrality", "community",
            "community_label", "in_lcc",
        ),
        "edge": (
            "paper_count", "weight", "fractional_weight", "record_ids",
            "years",
        ),
    },
    "exploration": {
        "node": (
            "label", "paper_count", "degree", "weighted_degree",
            "fractional_degree", "betweenness_centrality", "community",
            "community_label", "community_color", "in_lcc", "record_ids", "years",
            "representative_titles", "top_institutions", "top_countries",
            "top_terms", "topic_label",
        ),
        "edge": (
            "paper_count", "weight", "fractional_weight", "record_ids",
            "paper_ids", "titles", "years", "source_titles",
            "document_types", "keywords", "institutions", "countries",
            "top_terms", "topic_label",
        ),
    },
    "full": {
        "node": (
            "label", "paper_count", "degree", "degree_normalized",
            "weighted_degree", "fractional_degree", "betweenness_centrality",
            "community", "community_label", "in_lcc", "record_ids", "years",
            "community_color",
            "representative_titles", "source_titles", "document_types",
            "keywords", "top_institutions", "top_countries", "top_terms",
            "topic_label",
        ),
        "edge": (
            "paper_count", "weight", "fractional_weight", "record_ids",
            "paper_ids", "titles", "years", "source_titles",
            "document_types", "keywords", "institutions", "countries",
            "top_terms", "topic_label",
        ),
    },
}

SUPPORTED_NODE_ATTRIBUTES = frozenset(PROFILE_ATTRIBUTES["full"]["node"])
SUPPORTED_EDGE_ATTRIBUTES = frozenset(PROFILE_ATTRIBUTES["full"]["edge"])
COMMUNITY_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


@dataclass(frozen=True)
class GraphExportOptions:
    """Configuration for a rich graph export.

    For ``profile='custom'``, ``node_attributes`` and ``edge_attributes`` are
    used exactly.  For a named profile, non-empty custom lists override the
    corresponding profile list.  Edge titles are not truncated by default so
    that every link remains traceable to its supporting publications.
    """

    profile: str = "exploration"
    node_attributes: tuple[str, ...] = ()
    edge_attributes: tuple[str, ...] = ()
    top_terms: int = 8
    topic_label_terms: int = 3
    max_node_titles: int = 8
    max_edge_titles: int | None = None
    include_bigrams: bool = True
    text_source: str = "title_keywords"
    seed: int = 42
    community_resolution: float = 1.0
    betweenness_exact_max_nodes: int = 1000
    betweenness_samples: int = 250

    def __post_init__(self):
        if self.text_source not in {"title", "title_keywords"}:
            raise ValueError("text_source must be 'title' or 'title_keywords'")
        if self.top_terms < 0 or self.topic_label_terms < 0:
            raise ValueError("term limits must be non-negative")
        if self.max_node_titles is not None and self.max_node_titles < 0:
            raise ValueError("max_node_titles must be non-negative or None")
        if self.max_edge_titles is not None and self.max_edge_titles < 0:
            raise ValueError("max_edge_titles must be non-negative or None")

    def selected_attributes(self) -> tuple[set[str], set[str]]:
        profile = self.profile.lower()
        if profile == "custom":
            if not self.node_attributes and not self.edge_attributes:
                raise ValueError("custom profile requires node_attributes and/or edge_attributes")
            nodes, edges = set(self.node_attributes), set(self.edge_attributes)
        else:
            if profile not in PROFILE_ATTRIBUTES:
                raise ValueError(f"Unknown graph export profile: {self.profile}")
            base = PROFILE_ATTRIBUTES[profile]
            nodes = set(self.node_attributes or base["node"])
            edges = set(self.edge_attributes or base["edge"])
        unknown_nodes = sorted(nodes - SUPPORTED_NODE_ATTRIBUTES)
        unknown_edges = sorted(edges - SUPPORTED_EDGE_ATTRIBUTES)
        if unknown_nodes or unknown_edges:
            raise ValueError(
                f"Unsupported graph attributes; node={unknown_nodes}, edge={unknown_edges}"
            )
        return nodes, edges


def _clean_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value or "").split())


def _row_title(row: pd.Series) -> str:
    return _clean_text(row.get("title_one_line", "")) or _clean_text(row.get("title", ""))


def _row_record(row: pd.Series) -> dict[str, object]:
    rid = _clean_text(row.get("record_id", "")) or _clean_text(row.get("ut", ""))
    return {
        "record_id": rid,
        "year": int(row.get("year")) if pd.notna(row.get("year")) else None,
        "title": _row_title(row),
        "source_title": _clean_text(row.get("source_title", "")),
        "document_types": parse_list_cell(row.get("document_types", row.get("document_type_raw", ""))),
        "keywords": parse_list_cell(row.get("keywords", [])) + parse_list_cell(row.get("keywords_plus", [])),
        "institutions": parse_list_cell(row.get("institutions_normalized", row.get("institutions", []))),
        "countries": parse_list_cell(row.get("countries_normalized", row.get("countries", []))),
    }


def _unique(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(_clean_text(v) for v in values if _clean_text(v)))


def _record_sort_key(record: Mapping[str, object]) -> tuple[int, str, str]:
    year = record.get("year")
    return (int(year) if year is not None else 9999, str(record.get("record_id", "")), str(record.get("title", "")))


def _deduplicate_records(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[object, object, object]] = set()
    out = []
    for record in sorted(records, key=_record_sort_key):
        key = (record.get("record_id"), record.get("year"), record.get("title"))
        if key not in seen:
            seen.add(key)
            out.append(dict(record))
    return out


def _terms_for_records(
    records: Sequence[Mapping[str, object]],
    exclusions: set[str],
    merge_map: Mapping[str, str],
    include_bigrams: bool,
    text_source: str,
) -> Counter:
    counts: Counter = Counter()
    for record in records:
        tokens = tokenize_title(record.get("title", ""), exclusions, merge_map)
        counts.update(tokens)
        if include_bigrams:
            counts.update(" ".join(tokens[i:i + 2]) for i in range(max(0, len(tokens) - 1)))
        if text_source == "title_keywords":
            for keyword in record.get("keywords", []) or []:
                ktokens = tokenize_title(keyword, exclusions, merge_map)
                counts.update(ktokens)
                if include_bigrams:
                    counts.update(" ".join(ktokens[i:i + 2]) for i in range(max(0, len(ktokens) - 1)))
    return counts


def _ranked_values(values: Iterable[str], limit: int | None = None) -> list[str]:
    ranked = [value for value, _ in sorted(Counter(values).items(), key=lambda item: (-item[1], item[0]))]
    return ranked if limit is None else ranked[:limit]


def _record_attributes(
    records: Sequence[Mapping[str, object]],
    options: GraphExportOptions,
    exclusions: set[str],
    merge_map: Mapping[str, str],
    title_limit: int | None,
) -> dict[str, object]:
    records = _deduplicate_records(records)
    terms = _terms_for_records(records, exclusions, merge_map, options.include_bigrams, options.text_source)
    ranked_terms = [term for term, _ in sorted(terms.items(), key=lambda item: (-item[1], item[0]))]
    titles = _unique(record.get("title", "") for record in records)
    if title_limit is not None:
        titles = titles[:title_limit]
    return {
        "record_ids": _unique(record.get("record_id", "") for record in records),
        "paper_ids": _unique(record.get("record_id", "") for record in records),
        "titles": titles,
        "years": sorted({int(record["year"]) for record in records if record.get("year") is not None}),
        "source_titles": _ranked_values(
            source for record in records for source in [str(record.get("source_title", ""))] if source
        ),
        "document_types": _ranked_values(
            value for record in records for value in (record.get("document_types", []) or [])
        ),
        "keywords": _ranked_values(
            value for record in records for value in (record.get("keywords", []) or [])
        ),
        "institutions": _ranked_values(
            value for record in records for value in (record.get("institutions", []) or [])
        ),
        "countries": _ranked_values(
            value for record in records for value in (record.get("countries", []) or [])
        ),
        "top_terms": ranked_terms[: options.top_terms],
        "topic_label": ", ".join(ranked_terms[: options.topic_label_terms]),
    }


def _betweenness(G: nx.Graph, options: GraphExportOptions) -> tuple[dict, str]:
    n = G.number_of_nodes()
    if n == 0:
        return {}, "not_applicable"
    if n <= options.betweenness_exact_max_nodes:
        return nx.betweenness_centrality(G, normalized=True, endpoints=False, weight=None), "exact_unweighted_normalized"
    k = min(options.betweenness_samples, n)
    return (
        nx.betweenness_centrality(G, k=k, normalized=True, endpoints=False, weight=None, seed=options.seed),
        f"approx_unweighted_normalized_k{k}",
    )


def enrich_projection_graph(
    graph: nx.Graph,
    publications: pd.DataFrame,
    entities_col: str = "authors_network",
    options: GraphExportOptions | None = None,
    title_exclusions: Iterable[str] = (),
    title_merge_map: Mapping[str, str] | None = None,
    graph_kind: str = "author",
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[nx.Graph, dict[str, object]]:
    """Return a rich copy of a projection graph and an export audit.

    Publication metadata are aggregated separately for each node and for each
    pair of entities.  Therefore an edge's titles are exactly the publications
    that created that link, rather than all publications by either endpoint.
    """
    options = options or GraphExportOptions()
    node_selected, edge_selected = options.selected_attributes()
    exclusions = set(title_exclusions)
    merge_map = dict(title_merge_map or {})
    H = graph.copy()
    lcc_nodes = set(largest_component_graph(H).nodes()) if H.number_of_nodes() else set()
    node_records: dict[object, list[dict[str, object]]] = defaultdict(list)
    edge_records: dict[frozenset, list[dict[str, object]]] = defaultdict(list)

    for _, row in publications.iterrows():
        entities = list(dict.fromkeys(parse_list_cell(row.get(entities_col, []))))
        entities = [entity for entity in entities if entity in H]
        if not entities:
            continue
        record = _row_record(row)
        for entity in entities:
            node_records[entity].append(record)
        for left, right in combinations(entities, 2):
            if H.has_edge(left, right):
                edge_records[frozenset((left, right))].append(record)

    bc, bc_method = _betweenness(H, options)
    partition = louvain_partition(
        H.subgraph(lcc_nodes).copy(), seed=options.seed, weight="paper_count",
        resolution=options.community_resolution,
    ) if lcc_nodes else {}
    n = H.number_of_nodes()

    for node in H.nodes():
        record_attrs = _record_attributes(
            node_records.get(node, []), options, exclusions, merge_map,
            options.max_node_titles,
        )
        available = {
            "label": str(node),
            "paper_count": int(H.nodes[node].get("paper_count", len(_deduplicate_records(node_records.get(node, []))))),
            "degree": int(H.degree(node)),
            "degree_normalized": float(H.degree(node) / (n - 1)) if n > 1 else 0.0,
            "weighted_degree": float(H.degree(node, weight="weight")),
            "fractional_degree": float(H.degree(node, weight="fractional_weight")),
            "betweenness_centrality": float(bc.get(node, 0.0)),
            "community": int(partition.get(node, -1)),
            "community_label": f"C{partition[node]}" if node in partition else "outside_lcc",
            "community_color": COMMUNITY_COLORS[partition[node] % len(COMMUNITY_COLORS)] if node in partition else "#bdbdbd",
            "in_lcc": int(node in lcc_nodes),
            "record_ids": record_attrs["record_ids"],
            "years": record_attrs["years"],
            "representative_titles": record_attrs["titles"],
            "source_titles": record_attrs["source_titles"],
            "document_types": record_attrs["document_types"],
            "keywords": record_attrs["keywords"],
            "top_institutions": record_attrs["institutions"],
            "top_countries": record_attrs["countries"],
            "top_terms": record_attrs["top_terms"],
            "topic_label": record_attrs["topic_label"],
        }
        H.nodes[node].clear()
        H.nodes[node].update({key: value for key, value in available.items() if key in node_selected})

    unsupported_edges = []
    count_mismatches = []
    for left, right in H.edges():
        original = dict(graph[left][right])
        records = _deduplicate_records(edge_records.get(frozenset((left, right)), []))
        record_attrs = _record_attributes(
            records, options, exclusions, merge_map, options.max_edge_titles,
        )
        available = {
            "paper_count": int(original.get("paper_count", original.get("weight", len(records)))),
            "weight": float(original.get("weight", original.get("paper_count", len(records)))),
            "fractional_weight": float(original.get("fractional_weight", 0.0)),
            **record_attrs,
        }
        if not records:
            unsupported_edges.append((str(left), str(right)))
        if records and int(available["paper_count"]) != len(records):
            count_mismatches.append((str(left), str(right), int(available["paper_count"]), len(records)))
        H[left][right].clear()
        H[left][right].update({key: value for key, value in available.items() if key in edge_selected})

    H.graph.update({
        "graph_kind": graph_kind,
        "start_year": int(start_year) if start_year is not None else "",
        "end_year": int(end_year) if end_year is not None else "",
        "export_profile": options.profile,
        "node_attributes": sorted(node_selected),
        "edge_attributes": sorted(edge_selected),
        "community_method": "louvain_lcc",
        "community_seed": int(options.seed),
        "community_resolution": float(options.community_resolution),
        "centrality_measure": "betweenness_unweighted_normalized",
        "betweenness_method": bc_method,
        "text_summary_method": "frequency_ranked_title_keyword_unigrams_bigrams" if options.include_bigrams else "frequency_ranked_title_keyword_unigrams",
        "text_summary_status": "descriptive_exploration_not_topic_model",
        "entities_column": entities_col,
    })
    audit = {
        "graph_kind": graph_kind,
        "nodes": H.number_of_nodes(),
        "edges": H.number_of_edges(),
        "publication_rows": int(len(publications)),
        "profile": options.profile,
        "node_attributes": sorted(node_selected),
        "edge_attributes": sorted(edge_selected),
        "unsupported_edges": unsupported_edges,
        "paper_count_mismatches": count_mismatches,
        "betweenness_method": bc_method,
        "edge_titles_truncated": options.max_edge_titles is not None,
    }
    return H, audit


def _graphml_scalar(value: object) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return "" if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def graphml_safe_copy(graph: nx.Graph) -> nx.Graph:
    """Convert all graph, node, and edge values to GraphML-safe scalars."""
    H = graph.copy()
    H.graph = {key: _graphml_scalar(value) for key, value in H.graph.items()}
    for _, attributes in H.nodes(data=True):
        for key, value in list(attributes.items()):
            attributes[key] = _graphml_scalar(value)
    for _, _, attributes in H.edges(data=True):
        for key, value in list(attributes.items()):
            attributes[key] = _graphml_scalar(value)
    return H


def write_graphml_bundle(
    graph: nx.Graph,
    output_base: str | Path,
    formats: Sequence[str] = ("graphml", "graphmlz"),
) -> list[Path]:
    """Write standard GraphML and/or gzip-compressed GraphMLZ files.

    The graph remains semantically undirected.  ``.graphmlz`` is a gzip stream
    containing standard GraphML and can be opened by applications that support
    compressed GraphML, including Visone after decompression when necessary.
    """
    base = Path(output_base)
    if base.suffix.lower() in {".graphml", ".graphmlz"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    selected = [fmt.lower().lstrip(".") for fmt in formats]
    unknown = sorted(set(selected) - {"graphml", "graphmlz"})
    if unknown:
        raise ValueError(f"Unsupported graph export formats: {unknown}")
    H = graphml_safe_copy(graph)
    written: list[Path] = []
    graphml_path = base.with_suffix(".graphml")
    if "graphml" in selected:
        nx.write_graphml(H, graphml_path)
        written.append(graphml_path)
    if "graphmlz" in selected:
        graphmlz_path = base.with_suffix(".graphmlz")
        if graphml_path.exists():
            source = graphml_path
            temporary = None
        else:
            temporary = tempfile.NamedTemporaryFile(suffix=".graphml", delete=False)
            temporary.close()
            source = Path(temporary.name)
            nx.write_graphml(H, source)
        with source.open("rb") as src, gzip.open(graphmlz_path, "wb", compresslevel=9) as dst:
            dst.write(src.read())
        if temporary is not None:
            source.unlink(missing_ok=True)
        written.append(graphmlz_path)
    return written


def write_export_audit(audit: Mapping[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(audit), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
