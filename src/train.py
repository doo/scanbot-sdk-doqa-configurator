import base64
import gzip
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import CharacterClusteringTransformer
import click
import OpenCVSVMClassifier
import pandas as pd
import scanbotsdk
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from train_plot import plot_grid_search
from train_utils import all_features, best_low_complexity, get_character_properties


def process_image(
    file,
    dir_name,
    document_quality_analyzer: scanbotsdk.DocumentQualityAnalyzerTrainingDataAnnotator,
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
            label=1 if dir_name == 'good' else 0,
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


@click.command(context_settings={'show_default': True})
@click.option('--scanbotsdk_license_key', type=str, required=True, help='Scanbot SDK license key')
@click.option(
    '--training_dir',
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=Path(__file__).parent.parent / "data",
    help='Directory containing training images in the subfolders "good" and "bad"',
)
@click.option('--num_jobs', type=int, default=4, help='Number of parallel jobs for training')
@click.option('--smoke_test', is_flag=True, help='Run a smoke test with a small subset of the data')
@click.option(
    '--plot', is_flag=True, help='Create a plot of the grid search results', default=False
)
def main(
    scanbotsdk_license_key: str, training_dir: Path, num_jobs: int, smoke_test: bool, plot: bool
):
    scanbotsdk.initialize(scanbotsdk_license_key)
    document_quality_analyzer = scanbotsdk.DocumentQualityAnalyzerTrainingDataAnnotator()

    samples = []

    all_files = []
    for dir_name in ['good', 'bad']:
        dir = training_dir / dir_name
        file_extensions = ['png', 'jpg', 'jpeg']
        files = [file for ext in file_extensions for file in dir.glob(f'*.{ext}')]
        if smoke_test:
            files = files[:10]

        for file in files:
            all_files.append((file, dir_name))

    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        process_func = partial(process_image, document_quality_analyzer=document_quality_analyzer)
        future_to_file = {
            executor.submit(process_func, file, dir_name): (file, dir_name)
            for file, dir_name in all_files
        }

        for future in tqdm(
            as_completed(future_to_file), total=len(all_files), desc="Processing images"
        ):
            sample = future.result()
            if sample is not None:
                samples.append(sample)

    if len(samples) == 0:
        raise ValueError("No samples found")

    api_version = samples[0]['api_version'] if samples else None
    print(f"API version: {api_version}")

    random.shuffle(samples)

    labels = [s['label'] for s in samples]
    if 0 not in labels or 1 not in labels:
        raise ValueError(f"Missing samples from one class. Found labels: {set(labels)}")

    pipeline = Pipeline(
        [
            ('clustering', CharacterClusteringTransformer.CharacterClusteringTransformer()),
            ('svm', OpenCVSVMClassifier.OpenCVSVMClassifier()),
        ]
    )

    X = pd.DataFrame(samples)
    y = pd.Series([sample['label'] for sample in samples])

    num_positive = sum(y)
    num_negative = len(y) - num_positive
    if num_positive < 5 or num_negative < 5:
        raise ValueError(
            f"Not enough samples for training. \n"
            f"Please provide at least 5 samples for each class. \n"
            f"Positive: {num_positive}, Negative: {num_negative}"
        )

    scoring = {
        "accuracy": "balanced_accuracy",
        "num_support_vectors": lambda estimator, X, y: estimator.named_steps[
            'svm'
        ].get_num_support_vectors(),
    }

    param_grid = {
        "clustering__n_clusters": [6, 8, 10, 12, *range(15, 50, 5)],
        "clustering__cluster_features": [
            ["Contrast", "Ocrability", "FontSize"],
            ["Contrast", "Ocrability", "FontSize", "OrientationDeviation"],
        ],
        "svm__C": [1.0, 5.0, 10.0, 20.0, 40.0, 80.0],
        "svm__gamma_factor": [0.5, 1.0, 2.0, 4.0, 8.0],
        "svm__kernel": ['rbf'],
    }
    if smoke_test:
        param_grid["clustering__n_clusters"] = list(range(6, 10, 15))
        param_grid["svm__C"] = [1.0]

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=10,
        n_jobs=num_jobs,
        refit=best_low_complexity,
        error_score="raise",
        scoring=scoring,
        return_train_score=True,
    )
    clf.fit(X, y)

    if plot:
        plot_grid_search(clf, training_dir)

    # Save the model
    model_path = training_dir / 'DoQA_config.txt'
    with open(model_path, "w") as f:
        config = dict(
            type='SVM',
            api_version=api_version,
            **clf.best_estimator_.named_steps['svm'].export(),
            **clf.best_estimator_.named_steps['clustering'].export(),
        )
        json_string = json.dumps(config, indent=None, separators=(',', ': '))
        gzipped = gzip.compress(json_string.encode('utf-8'))
        b64_encoded = base64.b64encode(gzipped).decode('utf-8')
        f.write(b64_encoded)

    print(f"DoQA config saved to {model_path}")
    print(f"Cross-validation accuracy: {clf.cv_results_['mean_test_accuracy'][clf.best_index_]}")
    print(f"Parameters: {clf.best_params_}")
    print(
        f"Number of support vectors: {clf.best_estimator_.named_steps['svm'].get_num_support_vectors()}"
    )

    pred = clf.predict(X)
    predictions_report = []
    for i, sample in enumerate(samples):
        prediction = dict(
            prediction=int(pred[i]),
            filename=str(sample['image_path'].relative_to(training_dir)),
            label=sample['label'],
        )
        predictions_report.append(prediction)
    predictions_path = training_dir / 'predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions_report, f, indent=2)
    print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
