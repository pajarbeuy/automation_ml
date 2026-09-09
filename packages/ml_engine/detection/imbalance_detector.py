from __future__ import annotations

"""Class imbalance detector."""

from dataclasses import dataclass


@dataclass
class ImbalanceReport:
    minority_class: str
    minority_proportion: float
    majority_class: str
    majority_proportion: float
    imbalance_ratio: float
    is_imbalanced: bool  # True if ratio > 3:1


def detect_imbalance(class_distribution: dict[str, float]) -> ImbalanceReport | None:
    """
    Detect class imbalance from the class distribution dict.
    Returns None if only one class exists.
    """
    if len(class_distribution) < 2:
        return None

    sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1])
    minority_cls, minority_prop = sorted_classes[0]
    majority_cls, majority_prop = sorted_classes[-1]

    ratio = majority_prop / minority_prop if minority_prop > 0 else float("inf")

    return ImbalanceReport(
        minority_class=minority_cls,
        minority_proportion=minority_prop,
        majority_class=majority_cls,
        majority_proportion=majority_prop,
        imbalance_ratio=ratio,
        is_imbalanced=ratio > 3.0,
    )
