import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import papermill as pm
import scanbotsdk
from joblib import Memory
from nbconvert import HTMLExporter
from tqdm import tqdm

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


class ThresholdWaypoints(TypedDict):
    threshold_values_0p5: list[float]
    threshold_values_1p0: list[float]


def load_samples_from_training_dir(
    training_dir: str,
    smoke_test: bool,
    document_quality_analyzer: scanbotsdk.DocumentQualityAnalyzerTrainingDataAnnotator,
    num_jobs: int,
    cache_enabled: bool = True,
    show_progress: bool = True,
):
    memory = Memory(
        location=Path(training_dir) / "cache" / scanbotsdk.get_git_version()
        if cache_enabled
        else None,
        verbose=0,
    )

    @memory.cache()
    def process_image(
        file,
        label,
    ):
        """Process a single image file and return sample data."""
        try:
            image = scanbotsdk.ImageRef.from_path(file)
            result = document_quality_analyzer.run(image=image)
            character_level_annotations = result.character_level_annotations
            api_version = character_level_annotations.api_version

            if len(character_level_annotations.annotations) == 0:
                print(f"Skipping {file} because no characters could be detected")
                return None

            sample = dict(
                label=label,
                image_path=file,
                character_level_annotations=pd.DataFrame(
                    get_character_properties(character_level_annotations, all_features)
                ),
                api_version=api_version,
            )
            return sample
        except Exception as e:
            print(f"Error processing {file}: {e}")
            return None

    samples = []

    all_files = []
    for dir_name in ['good', 'bad']:
        dir = Path(training_dir) / dir_name
        files = [file for ext in image_extensions for file in dir.glob(f'*{ext}')]
        if smoke_test:
            files = files[:20]

        for file in files:
            all_files.append((file, dir_name))

    with (ThreadPoolExecutor(max_workers=num_jobs) as executor):
        process_func = partial(process_image)
        future_to_file = {
            executor.submit(process_func, file, 1 if dir_name == 'good' else 0): (file, dir_name)
            for file, dir_name in all_files
        }

        results = (
            tqdm(as_completed(future_to_file), total=len(all_files), desc="Processing images")
            if show_progress
            else as_completed(future_to_file)
        )
        for future in results:
            sample = future.result()
            if sample is not None:
                samples.append(sample)

    return samples


def render_notebook(notebook_path: Path, parameters: dict, output_path: Path):
    executed_notebook = pm.execute_notebook(
        notebook_path,
        None,
        parameters=parameters,
        cwd=Path(__file__).parent,
    )
    html_exporter = HTMLExporter()
    html_exporter.exclude_code_cell = False
    html_exporter.template_name = "lab"
    html_exporter.exclude_input = True
    html_exporter.preprocessors = ['nbconvert.preprocessors.TagRemovePreprocessor']
    html_exporter.embed_images = True
    (body, resources) = html_exporter.from_notebook_node(executed_notebook)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)


def pickle_dump(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
