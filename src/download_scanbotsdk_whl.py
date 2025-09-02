import sys
from pathlib import Path

import requests


def download_github_release_asset(
    repo_owner, repo_name, release_name, architecture, output_dir="."
):
    base_url = "https://api.github.com"

    releases_url = f"{base_url}/repos/{repo_owner}/{repo_name}/releases"
    response = requests.get(releases_url)
    response.raise_for_status()

    releases = response.json()
    target_release = None

    for release in releases:
        if release["name"] == release_name or release["tag_name"] == release_name:
            target_release = release
            break

    if not target_release:
        raise ValueError(f"Release '{release_name}' not found")

    target_asset = None
    for asset in target_release["assets"]:
        if asset["name"].endswith(architecture + ".whl"):
            target_asset = asset
            break

    if not target_asset:
        raise ValueError(
            f"Python package for architecure '{architecture}' not found in release '{release_name}'"
        )

    download_url = target_asset["browser_download_url"]
    file_size = target_asset["size"]

    output_path = Path(output_dir) / target_asset["name"]

    print(f"Downloading {target_asset['name']} ({file_size} bytes)...")

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Downloaded to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <release_name> <architecture>")
        sys.exit(1)

    release_name = sys.argv[1]
    architecture = sys.argv[2]

    try:
        download_github_release_asset(
            "doo", "scanbot-sdk-example-linux", release_name, architecture
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
