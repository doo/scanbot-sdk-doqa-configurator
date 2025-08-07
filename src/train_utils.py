import numpy as np
import scanbotsdk

all_features = [
    "Contrast",
    "Ocrability",
    "FontSize",
    "Orientation",
    "OrientationNormalized",
    "OrientationDeviation",
]


def get_character_properties(
    character_level_annotations: scanbotsdk.CharacterLevelAnnotations, cluster_features: list[str]
):
    results: list[dict] = []

    for character_level_annotation in character_level_annotations.annotations:
        result = {}
        for feature_name in cluster_features:
            if feature_name == "Contrast":
                value = character_level_annotation.contrast
            elif feature_name == "Ocrability":
                value = character_level_annotation.ocrability
            elif feature_name == "FontSize":
                value = character_level_annotation.font_size
            elif feature_name == "Orientation":
                value = character_level_annotation.orientation
            elif feature_name == "OrientationNormalized":
                value = character_level_annotation.orientation_normalized
            elif feature_name == "OrientationDeviation":
                value = character_level_annotation.orientation_deviation
            else:
                raise ValueError(f"Unknown feature name: {feature_name}")
            result[feature_name] = value
        results.append(result)

    return results


def best_low_complexity(cv_results):
    """
    Balance model complexity with cross-validated score.

    Idea for the future:
    Instead of this trade-off, we could use the following approach to compute the optimum number of clusters:
    Increase the number of clusters until there are too many clusters which are too poorly represented in the training set,
    e.g. more than 3 clusters with less than 100 characters or only characters from fewer than 3 documents.
    """
    best_accuracy = np.max(cv_results["mean_test_accuracy"])
    threshold = best_accuracy * 0.997
    candidate_idx = np.flatnonzero(cv_results["mean_test_accuracy"] >= threshold)
    best_idx = candidate_idx[cv_results["param_clustering__n_clusters"][candidate_idx].argmin()]
    return best_idx
