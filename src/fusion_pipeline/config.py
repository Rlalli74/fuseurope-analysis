"""Load and validate the final shared and article-specific method presets."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_PRESETS = ("shared", "guzzardi_lalli", "lalli")


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def load_method_preset(config_dir: str | Path, article: str) -> dict[str, Any]:
    """Return the shared configuration merged with one article preset."""
    if article not in {"guzzardi_lalli", "lalli"}:
        raise ValueError("article must be 'guzzardi_lalli' or 'lalli'")
    config_dir = Path(config_dir)
    shared = load_yaml(config_dir / "shared.yaml")
    specific = load_yaml(config_dir / f"{article}.yaml")
    return _deep_merge(shared, specific)


def validate_method_presets(config_dir: str | Path) -> dict[str, Any]:
    """Validate the minimum fields needed by the final two-paper workflow."""
    config_dir = Path(config_dir)
    loaded = {name: load_yaml(config_dir / f"{name}.yaml") for name in REQUIRED_PRESETS}
    shared = loaded["shared"]
    errors: list[str] = []
    if shared.get("corpora", {}).get("preserved_retrieval", {}).get("records") != 46443:
        errors.append("shared preserved retrieval count must be 46443")
    if shared.get("corpora", {}).get("analytical", {}).get("records") != 45453:
        errors.append("shared analytical count must be 45453")
    if shared.get("author_identity", {}).get("construct_on") != "preserved_retrieval":
        errors.append("author identity must be constructed on preserved_retrieval")
    gl = loaded["guzzardi_lalli"]
    if gl.get("network_snapshots") != [1979, 1986, 1996, 2001]:
        errors.append("Guzzardi-Lalli snapshots must be 1979, 1986, 1996, 2001")
    if gl.get("louvain", {}).get("seed") != 42:
        errors.append("Guzzardi-Lalli principal Louvain seed must be 42")
    la = loaded["lalli"]
    if la.get("country_reconstruction", {}).get("sources") != ["C1", "RP"]:
        errors.append("Lalli country reconstruction must use C1 and RP")
    if la.get("jet", {}).get("modified_representation") != "separate_JET_node":
        errors.append("Lalli modified representation must retain a separate JET node")
    if errors:
        raise ValueError("Invalid final method configuration: " + "; ".join(errors))
    return {"status": "pass", "presets": list(REQUIRED_PRESETS)}
