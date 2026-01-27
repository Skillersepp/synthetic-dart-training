"""
Custom Dataset mit On-The-Fly Background Augmentation

Lädt transparente Dartboard-Bilder und fügt zur Laufzeit
zufällige Hintergründe ein.
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Versuche Hugging Face datasets zu importieren
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class BackgroundProvider:
    """
    Stellt Hintergrundbilder bereit.
    Unterstützt: Lokaler Ordner, Hugging Face Datasets, oder Fallback-Farben.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_images: int = 10000,
        output_size: Tuple[int, int] = (800, 800)
    ):
        """
        Args:
            source: Entweder:
                    - Pfad zu lokalem Ordner mit Bildern
                    - "imagenet" / "coco" / "places365" für HuggingFace
                    - None für Fallback (zufällige Farben/Texturen)
            cache_dir: Cache-Ordner für heruntergeladene Datasets
            max_images: Maximale Anzahl Bilder die geladen werden
            output_size: Zielgröße (width, height)
        """
        self.output_size = output_size
        self.source = source
        self.images = []
        self.use_generated = False

        if source is None:
            print("Kein Hintergrund-Source angegeben, nutze generierte Texturen")
            self.use_generated = True

        elif os.path.isdir(source):
            # Lokaler Ordner
            self._load_from_directory(Path(source), max_images)

        elif HF_AVAILABLE and source.lower() in ['imagenet', 'coco', 'places365', 'textures']:
            # Hugging Face Dataset
            self._load_from_huggingface(source.lower(), cache_dir, max_images)

        else:
            print(f"Source '{source}' nicht gefunden, nutze generierte Texturen")
            self.use_generated = True

        if not self.use_generated:
            print(f"BackgroundProvider: {len(self.images)} Bilder geladen")

    def _load_from_directory(self, path: Path, max_images: int):
        """Lädt Bilder aus einem lokalen Ordner."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
        paths = []
        for ext in extensions:
            paths.extend(path.rglob(ext))

        paths = sorted(paths)[:max_images]
        self.images = [str(p) for p in paths]
        print(f"Geladen: {len(self.images)} Bilder aus {path}")

    def _load_from_huggingface(self, dataset_name: str, cache_dir: Optional[str], max_images: int):
        """Lädt Bilder von Hugging Face Datasets."""
        print(f"Lade {dataset_name} von Hugging Face...")

        dataset_configs = {
            'imagenet': ('imagenet-1k', 'train', 'image'),
            'coco': ('detection-datasets/coco', 'train', 'image'),
            'places365': ('ethz/food101', 'train', 'image'),  # Fallback - Places365 ist groß
            'textures': ('eugenesiow/dtd', 'train', 'image'),  # Describable Textures Dataset
        }

        if dataset_name not in dataset_configs:
            print(f"Unbekanntes Dataset: {dataset_name}")
            self.use_generated = True
            return

        try:
            ds_name, split, image_key = dataset_configs[dataset_name]

            # Streaming für große Datasets
            if dataset_name in ['imagenet', 'coco', 'places365']:
                ds = load_dataset(
                    ds_name,
                    split=split,
                    streaming=True,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                # Bei Streaming: Bilder on-demand laden
                self.streaming_dataset = ds
                self.image_key = image_key
                self.use_streaming = True
                print(f"Streaming-Modus aktiviert für {dataset_name}")
                return
            else:
                ds = load_dataset(
                    ds_name,
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )

            # Bilder extrahieren
            self.images = []
            for i, item in enumerate(ds):
                if i >= max_images:
                    break
                if image_key in item:
                    self.images.append(item[image_key])

            print(f"Geladen: {len(self.images)} Bilder von {ds_name}")

        except Exception as e:
            print(f"Fehler beim Laden von {dataset_name}: {e}")
            self.use_generated = True

    def _generate_background(self) -> np.ndarray:
        """Generiert einen zufälligen Hintergrund (Fallback)."""
        w, h = self.output_size
        bg_type = random.choice(['solid', 'gradient', 'noise', 'pattern'])

        if bg_type == 'solid':
            # Einfarbig
            color = [random.randint(0, 255) for _ in range(3)]
            bg = np.full((h, w, 3), color, dtype=np.uint8)

        elif bg_type == 'gradient':
            # Farbverlauf
            c1 = np.array([random.randint(0, 255) for _ in range(3)])
            c2 = np.array([random.randint(0, 255) for _ in range(3)])
            gradient = np.linspace(0, 1, h).reshape(-1, 1, 1)
            bg = ((1 - gradient) * c1 + gradient * c2).astype(np.uint8)
            bg = np.broadcast_to(bg, (h, w, 3)).copy()

        elif bg_type == 'noise':
            # Rauschen
            bg = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            bg = cv2.GaussianBlur(bg, (21, 21), 0)

        else:  # pattern
            # Einfaches Muster
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            color = [random.randint(50, 200) for _ in range(3)]
            step = random.randint(20, 50)
            for i in range(0, max(h, w), step):
                cv2.line(bg, (i, 0), (0, i), color, 1)
                cv2.line(bg, (w, i), (i, h), color, 1)

        return bg

    def _resize_and_crop(self, img: np.ndarray) -> np.ndarray:
        """Skaliert und croppt auf Zielgröße."""
        target_w, target_h = self.output_size
        h, w = img.shape[:2]

        # Skalierungsfaktor (Bild soll mindestens Zielgröße haben)
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Mittig croppen
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]

        return img

    def get_random(self) -> np.ndarray:
        """Gibt ein zufälliges Hintergrundbild zurück (BGR, uint8)."""
        if self.use_generated:
            return self._generate_background()

        if hasattr(self, 'use_streaming') and self.use_streaming:
            # Streaming-Modus: Bild on-demand laden
            try:
                item = next(iter(self.streaming_dataset.shuffle().take(1)))
                pil_img = item[self.image_key]
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                return self._resize_and_crop(img)
            except Exception:
                return self._generate_background()

        if len(self.images) == 0:
            return self._generate_background()

        # Zufälliges Bild auswählen
        choice = random.choice(self.images)

        if isinstance(choice, str):
            # Pfad zu Datei
            img = cv2.imread(choice)
            if img is None:
                return self._generate_background()
        else:
            # PIL Image (von HuggingFace)
            img = cv2.cvtColor(np.array(choice), cv2.COLOR_RGB2BGR)

        return self._resize_and_crop(img)


class DartboardDataset(Dataset):
    """
    Custom Dataset für Dartboard-Bilder mit On-The-Fly Background Augmentation.

    Kann direkt mit PyTorch DataLoader verwendet werden.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        background_source: Optional[str] = None,
        background_cache_dir: Optional[str] = None,
        img_size: int = 800,
        augment: bool = True
    ):
        """
        Args:
            images_dir: Ordner mit PNG-Bildern (mit Transparenz)
            labels_dir: Ordner mit YOLO-Labels (.txt)
            background_source: Pfad zu Hintergrundbildern oder Dataset-Name
            background_cache_dir: Cache für heruntergeladene Datasets
            img_size: Bildgröße
            augment: Augmentation aktivieren
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment

        # Bilder finden
        self.image_files = sorted(
            list(self.images_dir.glob('*.png')) +
            list(self.images_dir.glob('*.PNG'))
        )
        print(f"DartboardDataset: {len(self.image_files)} Bilder gefunden")

        # Background Provider
        self.bg_provider = BackgroundProvider(
            source=background_source,
            cache_dir=background_cache_dir,
            output_size=(img_size, img_size)
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        """
        Lädt ein Bild mit zufälligem Hintergrund.

        Returns:
            dict mit 'image' (Tensor) und 'labels' (Tensor)
        """
        img_path = self.image_files[idx]

        # Bild mit Alpha laden
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Konnte Bild nicht laden: {img_path}")

        # Auf Zielgröße skalieren
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))

        # Hintergrund-Compositing wenn Alpha-Kanal vorhanden
        if img.shape[2] == 4 and self.augment:
            bg = self.bg_provider.get_random()
            img = self._composite(img, bg)
        elif img.shape[2] == 4:
            # Ohne Augmentation: Weißer Hintergrund
            bg = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
            img = self._composite(img, bg)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Labels laden
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = self._load_labels(label_path)

        # Zu Tensor konvertieren
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels_tensor = torch.from_numpy(labels).float() if len(labels) > 0 else torch.zeros((0, 5))

        return {
            'image': img_tensor,
            'labels': labels_tensor,
            'path': str(img_path)
        }

    def _composite(self, foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Alpha-Blending von Vordergrund und Hintergrund."""
        alpha = foreground[:, :, 3:4] / 255.0
        fg_rgb = foreground[:, :, :3]
        blended = (fg_rgb * alpha + background * (1 - alpha)).astype(np.uint8)
        return blended

    def _load_labels(self, label_path: Path) -> np.ndarray:
        """Lädt YOLO-Format Labels."""
        if not label_path.exists():
            return np.zeros((0, 5))

        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts[:5]])

        return np.array(labels) if labels else np.zeros((0, 5))


def create_dataloader(
    images_dir: str,
    labels_dir: str,
    background_source: Optional[str] = None,
    batch_size: int = 16,
    img_size: int = 800,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = True
) -> torch.utils.data.DataLoader:
    """
    Erstellt einen DataLoader mit Background-Augmentation.

    Args:
        images_dir: Ordner mit PNG-Bildern
        labels_dir: Ordner mit YOLO-Labels
        background_source: Hintergrund-Quelle (Pfad oder Dataset-Name)
        batch_size: Batch-Größe
        img_size: Bildgröße
        shuffle: Daten mischen
        num_workers: Anzahl Worker-Prozesse
        augment: Augmentation aktivieren

    Returns:
        PyTorch DataLoader
    """
    dataset = DartboardDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        background_source=background_source,
        img_size=img_size,
        augment=augment
    )

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = [item['labels'] for item in batch]
        paths = [item['path'] for item in batch]
        return {'images': images, 'labels': labels, 'paths': paths}

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


# ============================================================
# Integration mit Ultralytics YOLO
# ============================================================

def patch_ultralytics_dataloader(
    background_source: Optional[str] = None,
    background_cache_dir: Optional[str] = None
):
    """
    Patcht den Ultralytics Dataloader für On-The-Fly Background Augmentation.

    EXPERIMENTELL: Ultralytics hat seinen eigenen Dataloader, dieser Patch
    versucht die Bilder nach dem Laden zu modifizieren.
    """
    print("HINWEIS: Für volle Kontrolle nutze den custom DataLoader")
    print("oder führe augment_backgrounds.py als Preprocessing aus.")


if __name__ == '__main__':
    # Test
    print("Testing BackgroundProvider...")

    # Test mit generierten Hintergründen
    provider = BackgroundProvider(source=None)
    for i in range(3):
        bg = provider.get_random()
        print(f"  Generated background {i}: shape={bg.shape}, dtype={bg.dtype}")
        cv2.imwrite(f"test_bg_{i}.png", bg)

    print("\nTest abgeschlossen. Siehe test_bg_*.png")
