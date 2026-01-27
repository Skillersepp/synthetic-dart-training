"""
Download DTD (Describable Textures Dataset) für Background Augmentation

Das DTD Dataset enthält ~5600 Texturbilder in 47 Kategorien.
Perfekt als Hintergrund für Dartboard-Training!

Quelle: https://www.robots.ox.ac.uk/~vgg/data/dtd/
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
    """Lädt eine Datei mit Fortschrittsanzeige herunter."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def download_dtd(output_dir: str = "datasets/backgrounds/dtd"):
    """
    Lädt das DTD Textures Dataset herunter.

    Args:
        output_dir: Zielordner für die Bilder
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prüfen ob bereits vorhanden
    images_dir = output_dir / "images"
    if images_dir.exists() and len(list(images_dir.rglob("*.jpg"))) > 1000:
        print(f"DTD bereits vorhanden in: {output_dir}")
        print(f"Gefunden: {len(list(images_dir.rglob('*.jpg')))} Bilder")
        return str(images_dir)

    # Download
    archive_path = output_dir / "dtd.tar.gz"
    print(f"Lade DTD Dataset herunter (~{DTD_SIZE_MB}MB)...")
    print(f"URL: {DTD_URL}")

    try:
        download_file(DTD_URL, archive_path)
    except Exception as e:
        print(f"Download fehlgeschlagen: {e}")
        print("\nAlternative: Manuell herunterladen von:")
        print(f"  {DTD_URL}")
        print(f"  und entpacken nach: {output_dir}")
        return None

    # Entpacken
    print("Entpacke Archiv...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(output_dir)

    # Aufräumen
    archive_path.unlink()

    # Bilder-Ordner finden
    images_dir = output_dir / "dtd" / "images"
    if not images_dir.exists():
        images_dir = output_dir / "images"

    n_images = len(list(images_dir.rglob("*.jpg")))
    print(f"\nFertig! {n_images} Texturbilder verfügbar in:")
    print(f"  {images_dir}")

    return str(images_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download DTD Textures Dataset")
    parser.add_argument(
        "--output", "-o",
        default="datasets/backgrounds/dtd",
        help="Zielordner"
    )
    args = parser.parse_args()

    # Relativ zum YOLO26 Ordner
    script_dir = Path(__file__).parent.parent
    output = script_dir / args.output

    result = download_dtd(str(output))
    if result:
        print(f"\nNutze diesen Pfad für --backgrounds:")
        print(f"  {result}")
