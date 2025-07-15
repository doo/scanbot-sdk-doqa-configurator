import json
import os
import random
from pathlib import Path
import click
import pandas as pd
import scanbotsdk
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import gzip
import base64

import CharacterClusteringTransformer
import OpenCVSVMClassifier


@click.command(context_settings={'show_default': True})
@click.option('--scanbotsdk_license_key', type=str, required=True, help='Scanbot SDK license key')
@click.option('--training_dir', type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
              default=Path(__file__).parent.parent / "data",
              help='Directory containing training images in the subfolders "good" and "bad"')
@click.option('--num_jobs', type=int, default=4, help='Number of parallel jobs for training')
@click.option('--smoke_test', is_flag=True, help='Run a smoke test with a small subset of the data')
def main(
        scanbotsdk_license_key: str,
        training_dir: Path,
        num_jobs: int,
        smoke_test: bool,
):
    scanbotsdk.initialize(scanbotsdk_license_key)
    document_quality_analyzer = scanbotsdk.DocumentQualityAnalyzer(
        configuration=scanbotsdk.DocumentQualityAnalyzerConfiguration(
            min_processed_fraction=1.0,
            max_processed_fraction=1.0,
        )
    )

    # Load images & run DQA
    samples = []
    for dir_name in ['good', 'bad']:
        dir = training_dir / dir_name
        file_extensions = ['png', 'jpg', 'jpeg']
        files = [file for ext in file_extensions for file in dir.glob(f'*.{ext}')]
        if smoke_test:
            files = files[:10]

        for file in tqdm(files, desc=f"Processing images in {dir}"):
            image = scanbotsdk.ImageRef.from_path(file)
            character_level_annotations = document_quality_analyzer.get_character_level_annotations(image=image)
            api_version = character_level_annotations.api_version
            sample = dict(
                label=1 if dir_name == 'good' else 0,
                image_path=file,
                character_level_annotations=character_level_annotations,
            )
            if len(character_level_annotations.annotations) == 0:
                print(f"Skipping {file} because no characters could be detected")
                continue
            samples.append(sample)

    if len(samples) == 0:
        raise ValueError("No samples found")

    labels = [s['label'] for s in samples]
    if 0 not in labels or 1 not in labels:
        raise ValueError(f"Missing samples from one class. Found labels: {set(labels)}")

    pipeline = Pipeline([
        ('clustering', CharacterClusteringTransformer.CharacterClusteringTransformer(n_clusters=10)),
        ('svm', OpenCVSVMClassifier.OpenCVSVMClassifier(kernel='rbf'))
    ])

    param_grid = {
        "clustering__n_clusters": range(8, 20),
        "svm__kernel": ['poly', 'rbf', 'sigmoid']
    }
    if smoke_test:
        param_grid = {
            "clustering__n_clusters": [10],
            "svm__kernel": ['rbf']
        }

    random.shuffle(samples)

    X = pd.DataFrame(samples)
    y = pd.Series([sample['label'] for sample in samples])

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        verbose=10,
        n_jobs=num_jobs,
        refit=True,
    )
    clf.fit(X, y)

    # Save the model
    model_path = training_dir / 'DoQA_config.txt'
    with open(model_path, "w") as f:
        config = dict(
            type='SVM',
            api_version=api_version,
            **clf.best_estimator_.named_steps['svm'].export(),
            **clf.best_estimator_.named_steps['clustering'].export()
        )
        json_string = json.dumps(config, indent=None, separators=(',', ': '))
        gzipped = gzip.compress(json_string.encode('utf-8'))
        b64_encoded = base64.b64encode(gzipped).decode('utf-8')
        f.write(b64_encoded)

    print(f"DoQA config saved to {model_path}")
    print(f"Cross-validation accuracy: {clf.best_score_}")
    print(f"Parameters: {clf.best_params_}")

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
