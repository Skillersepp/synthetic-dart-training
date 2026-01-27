"""
YOLO26 Training mit On-The-Fly Background Augmentation

Dieses Script patcht den Ultralytics Dataloader, sodass bei JEDEM
Trainingsschritt ein zufälliger Hintergrund eingefügt wird.

So hat jede Epoche für das gleiche Bild einen anderen Hintergrund!
"""

import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import os
import random
from typing import Optional, List
from tqdm import tqdm

import torch

try:
    from ultralytics import YOLO
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.augment import Compose, LetterBox
except ImportError:
    raise ImportError("ultralytics nicht installiert: pip install ultralytics")


class BackgroundCache:
    """
    Lädt und cached Hintergrundbilder für schnellen Zugriff.
    """
    _instance = None
    _backgrounds = []
    _initialized = False

    @classmethod
    def initialize(cls, source: str, max_images: int = 5000, target_size: int = 800):
        """Initialisiert den Background-Cache (Singleton)."""
        if cls._initialized:
            return

        print(f"Lade Hintergrundbilder aus: {source}")

        if os.path.isdir(source):
            # Lokaler Ordner
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            paths = []
            for ext in extensions:
                paths.extend(Path(source).rglob(ext))
            paths = sorted(paths)[:max_images]

            print(f"Lade {len(paths)} Bilder in RAM...")
            for p in tqdm(paths, desc="Loading backgrounds"):
                img = cv2.imread(str(p))
                if img is not None:
                    # Resize und crop auf Zielgröße
                    img = cls._resize_crop(img, target_size)
                    cls._backgrounds.append(img)

        else:
            # Generierte Hintergründe als Fallback
            print("Generiere zufällige Hintergründe...")
            for _ in tqdm(range(1000), desc="Generating backgrounds"):
                cls._backgrounds.append(cls._generate_background(target_size))

        print(f"Background-Cache: {len(cls._backgrounds)} Bilder geladen")
        cls._initialized = True

    @classmethod
    def _resize_crop(cls, img: np.ndarray, size: int) -> np.ndarray:
        """Resize und center-crop auf quadratische Größe."""
        h, w = img.shape[:2]
        scale = max(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        start_x = (new_w - size) // 2
        start_y = (new_h - size) // 2
        return img[start_y:start_y + size, start_x:start_x + size]

    @classmethod
    def _generate_background(cls, size: int) -> np.ndarray:
        """Generiert einen zufälligen Hintergrund."""
        bg_type = random.choice(['solid', 'gradient', 'noise'])

        if bg_type == 'solid':
            color = [random.randint(20, 235) for _ in range(3)]
            return np.full((size, size, 3), color, dtype=np.uint8)

        elif bg_type == 'gradient':
            c1 = np.array([random.randint(20, 235) for _ in range(3)])
            c2 = np.array([random.randint(20, 235) for _ in range(3)])
            gradient = np.linspace(0, 1, size).reshape(-1, 1, 1)
            bg = ((1 - gradient) * c1 + gradient * c2).astype(np.uint8)
            return np.broadcast_to(bg, (size, size, 3)).copy()

        else:  # noise
            bg = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
            return cv2.GaussianBlur(bg, (15, 15), 0)

    @classmethod
    def get_random(cls) -> np.ndarray:
        """Gibt ein zufälliges Hintergrundbild zurück."""
        if not cls._backgrounds:
            return cls._generate_background(800)
        return random.choice(cls._backgrounds).copy()


def apply_background_to_image(img: np.ndarray) -> np.ndarray:
    """
    Ersetzt transparenten Hintergrund durch zufälliges Bild.

    Args:
        img: BGRA oder BGR Bild

    Returns:
        BGR Bild ohne Transparenz
    """
    # Prüfen ob Alpha-Kanal vorhanden
    if img.shape[2] != 4:
        return img

    h, w = img.shape[:2]

    # Hintergrund holen und auf Bildgröße anpassen
    bg = BackgroundCache.get_random()
    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h))

    # Alpha-Blending
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    fg = img[:, :, :3].astype(np.float32)
    bg = bg.astype(np.float32)

    blended = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)
    return blended


# Patch für YOLO Dataset
_original_load_image = None
_original_cv2_imread = None


def patched_cv2_imread(path, flags=cv2.IMREAD_COLOR):
    """
    Gepatchte cv2.imread die automatisch IMREAD_UNCHANGED für PNGs nutzt.
    """
    global _original_cv2_imread

    # Für PNG-Dateien: Mit Alpha-Kanal laden
    if str(path).lower().endswith('.png'):
        img = _original_cv2_imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            # Hintergrund anwenden
            img = apply_background_to_image(img)
        return img

    return _original_cv2_imread(path, flags)


def patch_cv2_imread():
    """Patcht cv2.imread für transparente PNG-Unterstützung."""
    global _original_cv2_imread

    if _original_cv2_imread is not None:
        return  # Bereits gepatcht

    _original_cv2_imread = cv2.imread

    # cv2.imread global ersetzen
    cv2.imread = patched_cv2_imread
    print("cv2.imread gepatcht für transparente PNG-Bilder")


def train(
    dataset_yaml: str,
    backgrounds_dir: str,
    model_size: str = 'yolo26n',
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 800,
    device: str = '0',
    project: str = 'runs/train',
    name: str = None,
    **kwargs
):
    """
    Trainiert YOLO26 mit On-The-Fly Background Augmentation.
    """
    # Background-Cache initialisieren
    BackgroundCache.initialize(backgrounds_dir, target_size=imgsz)

    # cv2.imread patchen für transparente PNGs
    patch_cv2_imread()

    # Experiment-Name
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'dartboard_{model_size}_{timestamp}'

    print(f"\n{'='*60}")
    print(f"YOLO26 Training mit On-The-Fly Background Augmentation")
    print(f"{'='*60}")
    print(f"Modell:      {model_size}")
    print(f"Dataset:     {dataset_yaml}")
    print(f"Backgrounds: {backgrounds_dir}")
    print(f"Epochs:      {epochs}")
    print(f"Batch Size:  {batch_size}")
    print(f"Image Size:  {imgsz}")
    print(f"Device:      {device}")
    print(f"{'='*60}\n")

    # YOLO Modell laden
    model = YOLO(f'{model_size}.pt')

    # Training starten
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        # Geometrische Augmentations deaktiviert
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        # Leichte Farbvariation
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        # Sonstige
        plots=True,
        save=True,
        verbose=True,
        **kwargs
    )

    print(f"\n{'='*60}")
    print("Training abgeschlossen!")
    print(f"Best Model: {project}/{name}/weights/best.pt")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='YOLO26 Training mit On-The-Fly Background Augmentation'
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Pfad zur Dataset YAML'
    )
    parser.add_argument(
        '--backgrounds', '-b',
        type=str,
        required=True,
        help='Ordner mit Hintergrundbildern'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolo26n',
        help='Modell (yolo26n/s/m/l/x)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Anzahl Epochen'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch-Größe'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=800,
        help='Bildgröße'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device (0, cpu)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Output-Ordner'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment-Name'
    )

    args = parser.parse_args()

    train(
        dataset_yaml=args.data,
        backgrounds_dir=args.backgrounds,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()
