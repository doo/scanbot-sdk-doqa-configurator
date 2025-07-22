import scanbotsdk

all_features = [
    "Contrast", "Ocrability", "FontSize", "Orientation", "OrientationNormalized", "OrientationDeviation"
]

def get_character_properties(
        character_level_annotations: scanbotsdk.CharacterLevelAnnotations,
        cluster_features: [str]
):
    results: [dict] = []

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