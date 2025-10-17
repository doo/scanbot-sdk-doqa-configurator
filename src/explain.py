import os
from pathlib import Path

import click
from configurator_utils import image_extensions, render_notebook


@click.command(
    context_settings={'show_default': True},
    help="Generates explanation reports for all images in the 'explain' subfolder of the training directory. See the Readme.md for details.",
)
@click.option(
    '--scanbotsdk_license_key',
    type=str,
    required=True,
    help='Scanbot SDK license key',
    default=os.environ.get("SCANBOT_SDK_LICENSE", None),
)
@click.option(
    '--training_dir',
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
    default=Path(__file__).parent.parent / "data",
    help='Directory containing training images in the subfolders "good" and "bad". See the Readme.md for details.',
)
@click.option('--num_jobs', type=int, default=4, help='Number of parallel jobs for training')
def main(
    scanbotsdk_license_key: str,
    training_dir: Path,
    num_jobs: int,
):
    explain_dir: Path = training_dir / "explain"
    assert explain_dir.exists() and explain_dir.is_dir(), f"Directory {explain_dir} does not exist"
    assert (
        explain_dir.iterdir()
    ), f"Directory {explain_dir} is empty. Please add some images that should be explained."
    config_debug_path = training_dir / "DoQA_config_debug.pkl"
    assert config_debug_path.exists(), f"File {config_debug_path} does not exist"

    for file in explain_dir.iterdir():
        if file.suffix in image_extensions:
            render_notebook(
                notebook_path=Path(__file__).parent / "explain_report.ipynb",
                parameters=dict(
                    scanbotsdk_license_key=scanbotsdk_license_key,
                    training_dir=str(training_dir),
                    explain_image_path=str(file),
                    config_debug_path=str(config_debug_path),
                    num_jobs=num_jobs,
                ),
                output_path=explain_dir / f"report_{file.stem}.html",
            )


if __name__ == "__main__":
    main()
