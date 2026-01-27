"""
Custom Dataset with On-The-Fly Background Augmentation

Loads transparent dartboard images and inserts random backgrounds
at runtime.
"""

import os
import random
from pathlib import Path
from typing import Optional, List, Tuple, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Try to import Hugging Face datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class BackgroundProvider:
    """
    Provides background images.
    Supports: Local folder, Hugging Face Datasets, or fallback colors.
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
            source: Either:
                    - Path to local folder with images
                    - "imagenet" / "coco" / "places365" for HuggingFace
                    - None for fallback (random colors/textures)
            cache_dir: Cache directory for downloaded datasets
            max_images: Maximum number of images to load
            output_size: Target size (width, height)
        """
        self.output_size = output_size
        self.source = source
        self.images = []
        self.use_generated = False

        if source is None:
            print("No background source specified, using generated textures")
            self.use_generated = True

        elif os.path.isdir(source):
            # Local folder
            self._load_from_directory(Path(source), max_images)

        elif HF_AVAILABLE and source.lower() in ['imagenet', 'coco', 'places365', 'textures']:
            # Hugging Face Dataset
            self._load_from_huggingface(source.lower(), cache_dir, max_images)

        else:
            print(f"Source '{source}' not found, using generated textures")
            self.use_generated = True

        if not self.use_generated:
            print(f"BackgroundProvider: {len(self.images)} images loaded")

    def _load_from_directory(self, path: Path, max_images: int):
        """Loads images from a local folder."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
        paths = []
        for ext in extensions:
            paths.extend(path.rglob(ext))

        paths = sorted(paths)[:max_images]
        self.images = [str(p) for p in paths]
        print(f"Loaded: {len(self.images)} images from {path}")

    def _load_from_huggingface(self, dataset_name: str, cache_dir: Optional[str], max_images: int):
        """Loads images from Hugging Face Datasets."""
        print(f"Loading {dataset_name} from Hugging Face...")

        dataset_configs = {
            'imagenet': ('imagenet-1k', 'train', 'image'),
            'coco': ('detection-datasets/coco', 'train', 'image'),
            'places365': ('ethz/food101', 'train', 'image'),  # Fallback - Places365 is large
            'textures': ('eugenesiow/dtd', 'train', 'image'),  # Describable Textures Dataset
        }

        if dataset_name not in dataset_configs:
            print(f"Unknown dataset: {dataset_name}")
            self.use_generated = True
            return

        try:
            ds_name, split, image_key = dataset_configs[dataset_name]

            # Streaming for large datasets
            if dataset_name in ['imagenet', 'coco', 'places365']:
                ds = load_dataset(
                    ds_name,
                    split=split,
                    streaming=True,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                # For streaming: Load images on-demand
                self.streaming_dataset = ds
                self.image_key = image_key
                self.use_streaming = True
                print(f"Streaming mode enabled for {dataset_name}")
                return
            else:
                ds = load_dataset(
                    ds_name,
                    split=split,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )

            # Extract images
            self.images = []
            for i, item in enumerate(ds):
                if i >= max_images:
                    break
                if image_key in item:
                    self.images.append(item[image_key])

            print(f"Loaded: {len(self.images)} images from {ds_name}")

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            self.use_generated = True

    def _generate_background(self) -> np.ndarray:
        """Generates a random background (fallback)."""
        w, h = self.output_size
        bg_type = random.choice(['solid', 'gradient', 'noise', 'pattern'])

        if bg_type == 'solid':
            # Solid color
            color = [random.randint(0, 255) for _ in range(3)]
            bg = np.full((h, w, 3), color, dtype=np.uint8)

        elif bg_type == 'gradient':
            # Gradient
            c1 = np.array([random.randint(0, 255) for _ in range(3)])
            c2 = np.array([random.randint(0, 255) for _ in range(3)])
            gradient = np.linspace(0, 1, h).reshape(-1, 1, 1)
            bg = ((1 - gradient) * c1 + gradient * c2).astype(np.uint8)
            bg = np.broadcast_to(bg, (h, w, 3)).copy()

        elif bg_type == 'noise':
            # Noise
            bg = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            bg = cv2.GaussianBlur(bg, (21, 21), 0)

        else:  # pattern
            # Simple pattern
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            color = [random.randint(50, 200) for _ in range(3)]
            step = random.randint(20, 50)
            for i in range(0, max(h, w), step):
                cv2.line(bg, (i, 0), (0, i), color, 1)
                cv2.line(bg, (w, i), (i, h), color, 1)

        return bg

    def _resize_and_crop(self, img: np.ndarray) -> np.ndarray:
        """Scales and crops to target size."""
        target_w, target_h = self.output_size
        h, w = img.shape[:2]

        # Scaling factor (image must be at least target size)
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]

        return img

    def get_random(self) -> np.ndarray:
        """Returns a random background image (BGR, uint8)."""
        if self.use_generated:
            return self._generate_background()

        if hasattr(self, 'use_streaming') and self.use_streaming:
            # Streaming mode: Load image on-demand
            try:
                item = next(iter(self.streaming_dataset.shuffle().take(1)))
                pil_img = item[self.image_key]
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                return self._resize_and_crop(img)
            except Exception:
                return self._generate_background()

        if len(self.images) == 0:
            return self._generate_background()

        # Select random image
        choice = random.choice(self.images)

        if isinstance(choice, str):
            # Path to file
            img = cv2.imread(choice)
            if img is None:
                return self._generate_background()
        else:
            # PIL Image (from HuggingFace)
            img = cv2.cvtColor(np.array(choice), cv2.COLOR_RGB2BGR)

        return self._resize_and_crop(img)


class DartboardDataset(Dataset):
    """
    Custom Dataset for dartboard images with On-The-Fly Background Augmentation.

    Can be used directly with PyTorch DataLoader.
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
            images_dir: Folder with PNG images (with transparency)
            labels_dir: Folder with YOLO labels (.txt)
            background_source: Path to background images or dataset name
            background_cache_dir: Cache for downloaded datasets
            img_size: Image size
            augment: Enable augmentation
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment

        # Find images
        self.image_files = sorted(
            list(self.images_dir.glob('*.png')) +
            list(self.images_dir.glob('*.PNG'))
        )
        print(f"DartboardDataset: {len(self.image_files)} images found")

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
        Loads an image with a random background.

        Returns:
            dict with 'image' (Tensor) and 'labels' (Tensor)
        """
        img_path = self.image_files[idx]

        # Load image with alpha
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Scale to target size
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))

        # Background compositing if alpha channel exists
        if img.shape[2] == 4 and self.augment:
            bg = self.bg_provider.get_random()
            img = self._composite(img, bg)
        elif img.shape[2] == 4:
            # Without augmentation: White background
            bg = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
            img = self._composite(img, bg)

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Labels load
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = self._load_labels(label_path)

        # Convert to Tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels_tensor = torch.from_numpy(labels).float() if len(labels) > 0 else torch.zeros((0, 5))

        return {
            'image': img_tensor,
            'labels': labels_tensor,
            'path': str(img_path)
        }

    def _composite(self, foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Alpha blending of foreground and background."""
        alpha = foreground[:, :, 3:4] / 255.0
        fg_rgb = foreground[:, :, :3]
        blended = (fg_rgb * alpha + background * (1 - alpha)).astype(np.uint8)
        return blended

    def _load_labels(self, label_path: Path) -> np.ndarray:
        """Loads YOLO format labels."""
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
    Creates a DataLoader with background augmentation.

    Args:
        images_dir: Folder with PNG images
        labels_dir: Folder with YOLO labels
        background_source: Background source (path or dataset name)
        batch_size: Batch size
        img_size: Image size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        augment: Enable augmentation

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
# Integration with Ultralytics YOLO
# ============================================================

def patch_ultralytics_dataloader(
    background_source: Optional[str] = None,
    background_cache_dir: Optional[str] = None
):
    """
    Patches the Ultralytics Dataloader for On-The-Fly Background Augmentation.

    EXPERIMENTAL: Ultralytics has its own Dataloader, this patch
    attempts to modify the images after loading.
    """
    print("NOTE: Use the custom DataLoader for full control")
    print("or run augment_backgrounds.py as preprocessing.")


if __name__ == '__main__':
    # Test
    print("Testing BackgroundProvider...")

    # Test with generated backgrounds
    provider = BackgroundProvider(source=None)
    for i in range(3):
        bg = provider.get_random()
        print(f"  Generated background {i}: shape={bg.shape}, dtype={bg.dtype}")
        cv2.imwrite(f"test_bg_{i}.png", bg)

    print("\nTest finished. See test_bg_*.png")
