"""
YOLO26 Training with On-The-Fly Background Augmentation

This script patches the Ultralytics Dataloader so that a random
background is inserted at EVERY training step.

This ensures that each epoch sees a different background for the same image!
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
    raise ImportError("ultralytics not installed: pip install ultralytics")


class BackgroundCache:
    """
    Loads and caches background images for fast access.
    """
    _instance = None
    _backgrounds = []
    _initialized = False

    @classmethod
    def initialize(cls, source: str, max_images: int = 5000, target_size: int = 800):
        """Initializes the background cache (Singleton)."""
        if cls._initialized:
            return

        print(f"Loading background images from: {source}")

        if os.path.isdir(source):
            # Local folder
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
            paths = []
            for ext in extensions:
                paths.extend(Path(source).rglob(ext))
            paths = sorted(paths)[:max_images]

            print(f"Loading {len(paths)} images into RAM...")
            for p in tqdm(paths, desc="Loading backgrounds"):
                img = cv2.imread(str(p))
                if img is not None:
                    # Resize and crop to target size
                    img = cls._resize_crop(img, target_size)
                    cls._backgrounds.append(img)

        else:
            # Generated backgrounds as fallback
            print("Generating random backgrounds...")
            for _ in tqdm(range(1000), desc="Generating backgrounds"):
                cls._backgrounds.append(cls._generate_background(target_size))

        print(f"Background Cache: {len(cls._backgrounds)} images loaded")
        cls._initialized = True

    @classmethod
    def _resize_crop(cls, img: np.ndarray, size: int) -> np.ndarray:
        """Resize and center-crop to square size."""
        h, w = img.shape[:2]
        scale = max(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        start_x = (new_w - size) // 2
        start_y = (new_h - size) // 2
        return img[start_y:start_y + size, start_x:start_x + size]

    @classmethod
    def _generate_background(cls, size: int) -> np.ndarray:
        """Generates a random background."""
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
        """Returns a random background image."""
        if not cls._backgrounds:
            return cls._generate_background(800)
        return random.choice(cls._backgrounds).copy()


def apply_background_to_image(img: np.ndarray) -> np.ndarray:
    """
    Replaces transparent background with a random image.

    Args:
        img: BGRA or BGR image

    Returns:
        BGR image without transparency
    """
    # Check if Alpha channel exists
    if img.shape[2] != 4:
        return img

    h, w = img.shape[:2]

    # Get background and adapt to image size
    bg = BackgroundCache.get_random()
    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h))

    # Alpha-Blending
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    fg = img[:, :, :3].astype(np.float32)
    bg = bg.astype(np.float32)

    blended = (fg * alpha + bg * (1 - alpha)).astype(np.uint8)
    return blended


# Patch for YOLO Dataset
_original_load_image = None
_original_cv2_imread = None


def patched_cv2_imread(path, flags=cv2.IMREAD_COLOR):
    """
    Patched cv2.imread that automatically uses IMREAD_UNCHANGED for PNGs.
    """
    global _original_cv2_imread

    # For PNG files: Load with Alpha channel
    if str(path).lower().endswith('.png'):
        img = _original_cv2_imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            # Apply background
            img = apply_background_to_image(img)
        return img

    return _original_cv2_imread(path, flags)


def patch_cv2_imread():
    """Patches cv2.imread for transparent PNG support."""
    global _original_cv2_imread

    if _original_cv2_imread is not None:
        return  # Already patched

    _original_cv2_imread = cv2.imread

    # Replace cv2.imread globally
    cv2.imread = patched_cv2_imread
    print("cv2.imread patched for transparent PNG images")


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
    Trains YOLO26 with On-The-Fly Background Augmentation.
    """
    # Initialize Background Cache
    BackgroundCache.initialize(backgrounds_dir, target_size=imgsz)

    # Patch cv2.imread for transparent PNGs
    patch_cv2_imread()

    # Experiment Name
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'dartboard_{model_size}_{timestamp}'

    print(f"\n{'='*60}")
    print(f"YOLO26 Training with On-The-Fly Background Augmentation")
    print(f"{'='*60}")
    print(f"Model:       {model_size}")
    print(f"Dataset:     {dataset_yaml}")
    print(f"Backgrounds: {backgrounds_dir}")
    print(f"Epochs:      {epochs}")
    print(f"Batch Size:  {batch_size}")
    print(f"Image Size:  {imgsz}")
    print(f"Device:      {device}")
    print(f"{'='*60}\n")

    # Load YOLO Model
    model = YOLO(f'{model_size}.pt')

    # Start Training
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        # Geometric augmentations disabled
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        # Slight Color Variation
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        # Others
        plots=True,
        save=True,
        verbose=True,
        **kwargs
    )

    print(f"\n{'='*60}")
    print("Training finished!")
    print(f"Best Model: {project}/{name}/weights/best.pt")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='YOLO26 Training with On-The-Fly Background Augmentation'
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to Dataset YAML'
    )
    parser.add_argument(
        '--backgrounds', '-b',
        type=str,
        required=True,
        help='Folder with background images'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolo26n',
        help='Model (yolo26n/s/m/l/x)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=800,
        help='Image size'
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
