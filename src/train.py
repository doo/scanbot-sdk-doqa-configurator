import base64
import gzip
import json
import os
import random
import typing
from pathlib import Path

import CharacterClusteringTransformer
import click
import numpy as np
import OpenCVSVMClassifier
import pandas as pd
import scanbotsdk
from configurator_utils import (
    ThresholdWaypoints,
    best_low_complexity,
    load_samples_from_training_dir,
    pickle_dump,
    render_notebook,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from train_plots import plot_classification, plot_grid_search
from tsne_plot import tsne_plot


@click.command(context_settings={'show_default': True})
@click.option('--scanbotsdk_license_key', type=str, required=True, help='Scanbot SDK license key')
@click.option(
    '--training_dir',
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=Path(__file__).parent.parent / "data",
    help='Directory containing training images in the subfolders "good" and "bad". Please see the Readme.md for details.',
)
@click.option('--num_jobs', type=int, default=4, help='Number of parallel jobs for training')
@click.option('--smoke_test', is_flag=True, help='Run a smoke test with a small subset of the data')
@click.option(
    '--plot', is_flag=True, help='Create a plot of the grid search results', default=False
)
@click.option(
    '--cache',
    is_flag=True,
    help='Creates a folder "cache" and stores some intermediate results in there to speed up consecutive runs of this script',
    default=False,
)
def main(
    scanbotsdk_license_key: str,
    training_dir: Path,
    num_jobs: int,
    smoke_test: bool,
    plot: bool,
    cache: bool,
):
    scanbotsdk.set_logging(False)
    scanbotsdk.initialize(scanbotsdk_license_key)
    document_quality_analyzer = scanbotsdk.DocumentQualityAnalyzerTrainingDataAnnotator()

    samples = load_samples_from_training_dir(
        training_dir=training_dir,
        smoke_test=smoke_test,
        document_quality_analyzer=document_quality_analyzer,
        num_jobs=num_jobs,
        cache_enabled=cache,
        show_progress=True,
    )

    if len(samples) == 0:
        raise ValueError("No samples found")

    api_version = samples[0]['api_version'] if samples else None
    print(f"API version: {api_version}")
    print(f"Number of samples: {len(samples)}")

    random.shuffle(samples)

    labels = [s['label'] for s in samples]
    if 0 not in labels or 1 not in labels:
        raise ValueError(f"Missing samples from one class. Found labels: {set(labels)}")

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

    param_grid_clustering = {
        "clustering__n_clusters": [6, 8, 10, 12, *range(15, 50, 5)],
        "clustering__cluster_features": [
            ["Contrast", "Ocrability", "FontSize"],
            ["Contrast", "Ocrability", "FontSize", "OrientationDeviation"],
        ],
    }
    param_grid_svm = {
        "svm__C": [1.0, 5.0, 10.0, 20.0, 40.0, 80.0],
        "svm__gamma_factor": [0.5, 1.0, 2.0, 4.0, 8.0],
        "svm__kernel": ['rbf'],
    }
    if smoke_test:
        param_grid_clustering = {
            "clustering__n_clusters": [10],
            "clustering__cluster_features": [
                ["Contrast", "Ocrability", "FontSize", "OrientationDeviation"]
            ],
        }
        param_grid_svm = {
            "svm__C": [40],
            "svm__gamma_factor": [8],
            "svm__kernel": param_grid_svm["svm__kernel"],
        }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_inner = GridSearchCV(
        # We use a pipeline here to get the svm__ prefix so we don't confuse the parameters with those of the clustering step
        estimator=Pipeline(
            [
                ('svm', OpenCVSVMClassifier.OpenCVSVMClassifier()),
            ]
        ),
        param_grid=param_grid_svm,
        cv=cv,
        verbose=10,
        n_jobs=num_jobs,
        refit=False,
        error_score="raise",
        scoring=scoring,
        return_train_score=True,
    )

    joint_cv_results: typing.Any = None

    for clustering_params in ParameterGrid(param_grid_clustering):
        print(f"Evaluating clustering parameters: {clustering_params}")
        pipeline_clustering = Pipeline(
            [('clustering', CharacterClusteringTransformer.CharacterClusteringTransformer())]
        )
        X_transformed = pipeline_clustering.set_params(**clustering_params).fit_transform(X)
        clf_inner.fit(pd.DataFrame(X_transformed), y)
        for params in clf_inner.cv_results_['params']:
            params.update(clustering_params)
        inner_results = pd.DataFrame(clf_inner.cv_results_)
        for param, value in clustering_params.items():
            inner_results["param_" + param] = [value] * len(inner_results)
        joint_cv_results = (
            pd.concat([joint_cv_results, inner_results], ignore_index=True)
            if joint_cv_results is not None
            else inner_results
        )

    best_index = best_low_complexity(joint_cv_results)
    best_params = joint_cv_results['params'][best_index]
    pipeline = Pipeline(
        [
            ('clustering', CharacterClusteringTransformer.CharacterClusteringTransformer()),
            ('svm', OpenCVSVMClassifier.OpenCVSVMClassifier()),
        ]
    )
    pipeline.set_params(**best_params)

    y_cv_pred = pd.Series(index=y.index, dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        X_cv_train = X.iloc[train_idx]
        y_cv_train = y.iloc[train_idx]
        X_cv_test = X.iloc[test_idx]

        pipeline.fit(X_cv_train, y_cv_train)
        y_cv_pred[test_idx] = pipeline.decision_function(X_cv_test)

    median_split_index = np.argsort(
        [joint_cv_results[f'split{i}_test_accuracy'][best_index] for i in range(cv.get_n_splits())]
    )[cv.get_n_splits() // 2]
    train_idx, test_idx = list(cv.split(X, y))[median_split_index]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    pipeline.fit(X_train, y_train)

    if plot:
        plot_grid_search(joint_cv_results, training_dir)
        tsne_plot(training_dir=training_dir, pipeline=pipeline, X=X)

    threshold_waypoints = ThresholdWaypoints(
        threshold_values_0p5=[0.0, 0.0],
        threshold_values_1p0=[0.0, 0.0],
    )
    svm_decision_threshold = 0
    possible_threshold_points_pos = y_cv_pred[y_cv_pred >= svm_decision_threshold]
    possible_threshold_points_neg = y_cv_pred[y_cv_pred <= svm_decision_threshold]
    num_pos_classified_as_uncertain = np.asarray(
        [
            np.sum(
                (svm_decision_threshold < y_cv_pred) & (y_cv_pred <= possible_threshold) & (y == 1)
            )
            for possible_threshold in possible_threshold_points_pos
        ]
    )
    num_neg_classified_as_pos = np.asarray(
        [
            np.sum((y_cv_pred > possible_threshold) & (y == 0))
            for possible_threshold in possible_threshold_points_pos
        ]
    )
    num_neg_classified_as_uncertain = np.asarray(
        [
            np.sum(
                (possible_threshold <= y_cv_pred) & (y_cv_pred < svm_decision_threshold) & (y == 0)
            )
            for possible_threshold in possible_threshold_points_neg
        ]
    )
    num_pos_classified_as_neg = np.asarray(
        [
            np.sum((y_cv_pred < possible_threshold) & (y == 1))
            for possible_threshold in possible_threshold_points_neg
        ]
    )
    threshold_0p5_pos = np.argmin(
        np.abs(num_neg_classified_as_pos - num_pos_classified_as_uncertain)
    )
    threshold_0p5_neg = np.argmin(
        np.abs(num_pos_classified_as_neg - num_neg_classified_as_uncertain)
    )
    threshold_waypoints['threshold_values_0p5'][1] = max(
        possible_threshold_points_pos.iloc[threshold_0p5_pos],
        np.percentile(y_cv_pred[(y == 1) & (svm_decision_threshold < y_cv_pred)], 3),
    )
    threshold_waypoints["threshold_values_0p5"][0] = min(
        possible_threshold_points_neg.iloc[threshold_0p5_neg],
        np.percentile(y_cv_pred[(y == 0) & (y_cv_pred < svm_decision_threshold)], 97),
    )

    threshold_waypoints["threshold_values_1p0"][1] = np.percentile(
        y_cv_pred[(y == 1) & (y_cv_pred >= possible_threshold_points_pos.iloc[threshold_0p5_pos])],
        95,
    )
    threshold_waypoints["threshold_values_1p0"][0] = np.percentile(
        y_cv_pred[(y == 0) & (y_cv_pred <= possible_threshold_points_neg.iloc[threshold_0p5_neg])],
        5,
    )

    if plot:
        plot_classification(
            y=y,
            y_pred=y_cv_pred,
            threshold_waypoints=threshold_waypoints,
            output_dir=str(training_dir),
        )

    pred = pipeline.predict(X)
    pred_proba = pipeline.decision_function(X)
    predictions_report = []
    for i, sample in enumerate(samples):
        prediction = dict(
            prediction=int(pred[i]),
            prediction_proba=float(pred_proba[i]),
            filename=str(sample['image_path'].relative_to(training_dir)),
            label=sample['label'],
        )
        predictions_report.append(prediction)
    predictions_path = training_dir / 'predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions_report, f, indent=2)
    print(f"Predictions saved to {predictions_path}")

    model_path = training_dir / 'DoQA_config.txt'
    with open(model_path, "w") as f:
        config = dict(
            type='SVM',
            api_version=api_version,
            **pipeline.named_steps['svm'].export(),
            **pipeline.named_steps['clustering'].export(),
            threshold_normalization_points=threshold_waypoints,
        )
        json_string = json.dumps(config, indent=None, separators=(',', ': '))
        gzipped = gzip.compress(json_string.encode('utf-8'))
        b64_encoded = base64.b64encode(gzipped).decode('utf-8')
        f.write(b64_encoded)

    # This pickle file contains the pipeline object to be used in the explainability notebook (see Readme.md)
    pickle_dump(
        dict(clustering=pipeline.named_steps['clustering']),
        training_dir / 'DoQA_config_debug.pkl',
    )

    print(f"DoQA config saved to {model_path}")
    print(f"Cross-validation accuracy: {joint_cv_results['mean_test_accuracy'][best_index]}")
    print(f"Accuracy on training set: {pipeline.score(X_train, y_train)}")
    print(f"Parameters: {pipeline.get_params()['steps']}")
    print(f"Number of support vectors: {pipeline.named_steps['svm'].get_num_support_vectors()}")
    print(f"Threshold waypoints: {threshold_waypoints}")

    # This pickle file contains the training results to be used in the training report notebook
    pickle_file = training_dir / "DoQA_config.pkl"
    pickle_dump(
        dict(
            X=X[['image_path', 'label']],
            y=y,
            y_cv_pred=y_cv_pred,
            threshold_waypoints=threshold_waypoints,
            training_dir=str(training_dir),
            model_path=model_path,
            model_config=b64_encoded,
            mean_test_accuracy=joint_cv_results['mean_test_accuracy'][best_index],
        ),
        pickle_file,
    )
    training_report_path = training_dir / "training_report.html"
    render_notebook(
        notebook_path=Path(__file__).parent / "training_report.ipynb",
        parameters=dict(
            pickle_file=str(pickle_file),
        ),
        output_path=training_report_path,
    )
    print(f"Training report saved to {training_report_path}")
    os.unlink(pickle_file)


if __name__ == "__main__":
    main()
