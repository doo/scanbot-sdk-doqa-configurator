import os
from pathlib import Path

import click
import scanbotsdk


@click.command(
    context_settings={'show_default': True},
    help='Converts all PDF files in the "good" and "bad" subfolder of the training directory to PNG images and saves them in the respective subfolders.',
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
    help='Directory containing training images in the subfolders "good" and "bad". Please see the Readme.md for details.',
)
def pdf_to_png(
    training_dir: Path,
    scanbotsdk_license_key: str,
):
    scanbotsdk.initialize(license_key=scanbotsdk_license_key)
    multi_page_image_extractor = scanbotsdk.MultiPageImageExtractor()

    for subfolder in ['good', 'bad']:
        folder_path = training_dir / subfolder
        assert (
            folder_path.exists() and folder_path.is_dir()
        ), f"Directory {folder_path} does not exist"

        for pdf_file in folder_path.glob('*.pdf'):
            pages = multi_page_image_extractor.run(
                source=scanbotsdk.RandomAccessSource(file_path=pdf_file)
            )
            for i, page in enumerate(pages.pages):
                for j, image in enumerate(page.images):
                    output_path = folder_path / f"{pdf_file.stem}_page_{i + 1}_{j + 1}.png"
                    image.image.save_image(output_path)
                    print(f"Saved {output_path}")


if __name__ == "__main__":
    pdf_to_png()
