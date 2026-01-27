"""
Download DTD (Describable Textures Dataset) for background augmentation.

The DTD dataset contains ~5600 texture images in 47 categories.
Perfect as backgrounds for dartboard training!

Source: https://www.robots.ox.ac.uk/~vgg/data/dtd/
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_SIZE_MB = 600


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Downloads a file with a progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def download_dtd(output_dir: str = "datasets/backgrounds/dtd"):
    """
    Downloads the DTD Textures Dataset.

    Args:
        output_dir: Target folder for the images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    images_dir = output_dir / "images"
    if images_dir.exists() and len(list(images_dir.rglob("*.jpg"))) > 1000:
        print(f"DTD already exists in: {output_dir}")
        print(f"Found: {len(list(images_dir.rglob('*.jpg')))} images")
        return str(images_dir)

    # Download
    archive_path = output_dir / "dtd.tar.gz"
    print(f"Downloading DTD dataset (~{DTD_SIZE_MB}MB)...")
    print(f"URL: {DTD_URL}")

    try:
        download_file(DTD_URL, archive_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  {DTD_URL}")
        print(f"  and extract to: {output_dir}")
        return None

    # Extract
    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(output_dir)

    # Cleanup
    archive_path.unlink()

    # Find images folder
    images_dir = output_dir / "dtd" / "images"
    if not images_dir.exists():
        images_dir = output_dir / "images"

    n_images = len(list(images_dir.rglob("*.jpg")))
    print(f"\nDone! {n_images} texture images available in:")
    print(f"  {images_dir}")

    return str(images_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download DTD Textures Dataset")
    parser.add_argument(
        "--output", "-o",
        default="datasets/backgrounds/dtd",
        help="Target folder"
    )
    args = parser.parse_args()

    # Relative to YOLO26 folder
    script_dir = Path(__file__).parent.parent
    output = script_dir / args.output

    result = download_dtd(str(output))
    if result:
        print(f"\nNutze diesen Pfad für --backgrounds:")
        print(f"  {result}")
