from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from pathlib import Path
from typing import List


def compute_metrics(
    leakage_images_true: str, leakage_images_pred: List[Path], all_images: List[Path]
) -> dict:
    all_images = [img.name for img in all_images]
    leakage_images_pred = [img.name for img in leakage_images_pred]

    y_true = [1 if img in leakage_images_true else 0 for img in all_images]
    y_pred = [1 if img in leakage_images_pred else 0 for img in all_images]

    precision_score_value = precision_score(y_true, y_pred, zero_division=0)
    recall_score_value = recall_score(y_true, y_pred, zero_division=0)
    f1_score_value = f1_score(y_true, y_pred, zero_division=0)

    print(
        f"\
        Presicion: {precision_score_value:.4f}\n\
        Recall: {recall_score_value:.4f}\n\
        F1-score: {f1_score_value:.4f}"
    )

    return {
        "precision": precision_score_value,
        "recall": recall_score_value,
        "f1_score": f1_score_value,
    }