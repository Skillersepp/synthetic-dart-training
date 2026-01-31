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
import yaml
from typing import Optional, List
from tqdm import tqdm

import torch

try:
    from ultralytics import YOLO
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.augment import Compose, LetterBox
except ImportError:
    raise ImportError("ultralytics not installed: pip install ultralytics")


def load_config(config_path: Path) -> dict:
    """Loads the training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def apply_background_to_image(img: np.ndarray, noise_level: float = 0.0) -> np.ndarray:
    """
    Replaces transparent background with a random image.

    Args:
        img: BGRA or BGR image
        noise_level: Gaussian noise level (0.0 = none, 0.02 = 2% noise)

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

    blended = (fg * alpha + bg * (1 - alpha))

    # Add Gaussian noise if enabled
    if noise_level > 0:
        noise = np.random.randn(h, w, 3) * (noise_level * 255)
        blended = blended + noise

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


# Patch for YOLO Dataset
_original_load_image = None
_original_cv2_imread = None
_gaussian_noise_level = 0.0  # Global noise level for augmentation


def patched_cv2_imread(path, flags=cv2.IMREAD_COLOR):
    """
    Patched cv2.imread that automatically uses IMREAD_UNCHANGED for PNGs.
    """
    global _original_cv2_imread, _gaussian_noise_level

    # For PNG files: Load with Alpha channel
    if str(path).lower().endswith('.png'):
        img = _original_cv2_imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            # Apply background with noise
            img = apply_background_to_image(img, noise_level=_gaussian_noise_level)
        return img

    return _original_cv2_imread(path, flags)


def patch_cv2_imread(noise_level: float = 0.0):
    """
    Patches cv2.imread for transparent PNG support.

    Args:
        noise_level: Gaussian noise level (0.0 = none, 0.02 = 2%)
    """
    global _original_cv2_imread, _gaussian_noise_level

    if _original_cv2_imread is not None:
        return  # Already patched

    _gaussian_noise_level = noise_level
    _original_cv2_imread = cv2.imread

    # Replace cv2.imread globally
    cv2.imread = patched_cv2_imread
    print(f"cv2.imread patched for transparent PNG images (noise: {noise_level*100:.1f}%)")


def train(
    dataset_yaml: str,
    backgrounds_dir: str,
    config_path: str = None,
    model_size: str = None,
    epochs: int = None,
    batch_size: int = None,
    imgsz: int = None,
    device: str = None,
    workers: int = None,
    project: str = 'runs/train',
    name: str = None,
    **kwargs
):
    """
    Trains YOLO26 with On-The-Fly Background Augmentation.
    """
    # Load config if available
    config = {}
    if config_path and Path(config_path).exists():
        config = load_config(Path(config_path))
        print(f"Config loaded: {config_path}")

    # 1. Defaults
    default_epochs = 100
    default_batch = 16
    default_imgsz = 800
    default_device = '0'
    default_workers = 8
    default_model = 'yolo26n'

    # 2. Config overrides Defaults
    training_cfg = config.get('training', {})
    model_cfg = config.get('model', {})
    aug_cfg = config.get('augmentation', {})
    hw_cfg = config.get('hardware', {})

    cfg_epochs = training_cfg.get('epochs', default_epochs)
    cfg_batch = training_cfg.get('batch_size', default_batch)
    cfg_imgsz = training_cfg.get('imgsz', default_imgsz)
    cfg_model = model_cfg.get('name', default_model)
    cfg_device = hw_cfg.get('device', default_device)
    cfg_workers = hw_cfg.get('workers', default_workers)

    # 3. CLI Args override Config
    epochs = epochs if epochs is not None else cfg_epochs
    batch_size = batch_size if batch_size is not None else cfg_batch
    imgsz = imgsz if imgsz is not None else cfg_imgsz
    model_size = model_size if model_size is not None else cfg_model
    device = device if device is not None else cfg_device
    workers = workers if workers is not None else cfg_workers

    # Initialize Background Cache
    BackgroundCache.initialize(backgrounds_dir, target_size=imgsz)

    # Get custom augmentation settings
    custom_aug_cfg = config.get('custom_augmentation', {})
    noise_level = custom_aug_cfg.get('gaussian_noise', 0.0)

    # Patch cv2.imread for transparent PNGs with noise augmentation
    patch_cv2_imread(noise_level=noise_level)

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

    # Augmentation parameters from config or defaults
    augment_args = {
        'degrees': aug_cfg.get('degrees', 0.0),
        'translate': aug_cfg.get('translate', 0.1),
        'scale': aug_cfg.get('scale', 0.3),
        'shear': aug_cfg.get('shear', 0.0),
        'perspective': aug_cfg.get('perspective', 0.0003),
        'flipud': aug_cfg.get('flipud', 0.0),
        'fliplr': aug_cfg.get('fliplr', 0.0),
        'hsv_h': aug_cfg.get('hsv_h', 0.015),
        'hsv_s': aug_cfg.get('hsv_s', 0.4),
        'hsv_v': aug_cfg.get('hsv_v', 0.4),
        'bgr': aug_cfg.get('bgr', 0.0),
        'mosaic': aug_cfg.get('mosaic', 0.0),
        'mixup': aug_cfg.get('mixup', 0.0),
        'copy_paste': aug_cfg.get('copy_paste', 0.0),
        'cutmix': aug_cfg.get('cutmix', 0.0),
        'auto_augment': aug_cfg.get('auto_augment', ''),
        'erasing': aug_cfg.get('erasing', 0.1),
    }

    # Validation config
    val_cfg = config.get('validation', {})

    # Start Training
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        workers=workers,
        # Optimizer
        optimizer=training_cfg.get('optimizer', 'AdamW'),
        lr0=training_cfg.get('lr0', 0.001),
        lrf=training_cfg.get('lrf', 0.01),
        momentum=training_cfg.get('momentum', 0.937),
        weight_decay=training_cfg.get('weight_decay', 0.0005),
        warmup_epochs=training_cfg.get('warmup_epochs', 3),
        warmup_momentum=training_cfg.get('warmup_momentum', 0.8),
        warmup_bias_lr=training_cfg.get('warmup_bias_lr', 0.1),
        # Augmentations
        **augment_args,
        # Loss
        box=config.get('loss', {}).get('box', 7.5),
        cls=config.get('loss', {}).get('cls', 0.5),
        dfl=config.get('loss', {}).get('dfl', 1.5),
        # NMS & Detection (wichtig für dichte Darts!)
        iou=training_cfg.get('iou', val_cfg.get('iou_threshold', 0.4)),
        max_det=training_cfg.get('max_det', 300),
        conf=val_cfg.get('conf_threshold', 0.25),
        # Advanced Training
        close_mosaic=training_cfg.get('close_mosaic', 10),
        # Output
        plots=True,
        save=True,
        save_period=config.get('output', {}).get('save_period', 10),
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
        '--config', '-c',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training config'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model (yolo26n/s/m/l/x)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=None,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Image size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (0, cpu)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='DataLoader Workers'
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

    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / args.config if args.config else None

    train(
        dataset_yaml=args.data,
        backgrounds_dir=args.backgrounds,
        config_path=str(config_path) if config_path else None,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()
