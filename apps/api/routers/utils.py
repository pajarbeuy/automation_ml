"""Shared utilities for API routers."""
from __future__ import annotations

import sys
from pathlib import Path

_PKG = str(Path(__file__).parent.parent.parent / "packages" / "ml_engine")

if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from profiling.profiler import DatasetProfile, FeatureProfile


def features_from_profile_json(profile_json: dict) -> list[FeatureProfile]:
    """Reconstruct FeatureProfile list from stored profile JSON."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(FeatureProfile)}
    features = []
    for f_dict in profile_json.get("features", []):
        kwargs = {k: f_dict.get(k) for k in field_names if k in f_dict}
        # Ensure list field defaults
        if "top_categories" not in kwargs:
            kwargs["top_categories"] = []
        fp = FeatureProfile(**kwargs)
        fp.top_categories = f_dict.get("top_categories", [])
        features.append(fp)
    return features


def dataset_profile_from_json(profile_json: dict) -> DatasetProfile:
    """Reconstruct full DatasetProfile from stored JSON."""
    return DatasetProfile(
        rows=profile_json["rows"],
        columns=profile_json["columns"],
        target=profile_json["target"],
        duplicate_rows=profile_json["duplicate_rows"],
        features=features_from_profile_json(profile_json),
        class_distribution=profile_json["class_distribution"],
        dataset_hash=profile_json["dataset_hash"],
    )
