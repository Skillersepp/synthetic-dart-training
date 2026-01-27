"""
Background Augmentation

Replaces the transparent background of dartboard images with
random images from a background dataset.

The dartboard images must be PNGs with an alpha channel.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm
import shutil


class BackgroundAugmentor:
    """
    Adds random backgrounds to transparent dartboard images.
    """

    def __init__(
        self,
        background_dir: Path,
        output_size: Tuple[int, int] = (800, 800),
        offset_enabled: bool = False,
        max_offset_fraction: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            background_dir: Directory containing background images
            output_size: Output size (width, height)
            offset_enabled: Randomly shift dartboard (disabled by default)
            max_offset_fraction: Maximum offset as a fraction of image size
            seed: Random seed for reproducibility
        """
        self.background_dir = Path(background_dir)
        self.output_size = output_size
        self.offset_enabled = offset_enabled
        self.max_offset_fraction = max_offset_fraction

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load background images
        self.background_paths = self._find_backgrounds()
        print(f"Found: {len(self.background_paths)} background images")

        if len(self.background_paths) == 0:
            raise ValueError(
                f"No background images found in: {self.background_dir}\n"
                f"Supported formats: .jpg, .jpeg, .png, .webp"
            )

    def _find_backgrounds(self) -> List[Path]:
        """Finds all background images in the folder (recursively)."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
        paths = []
        for ext in extensions:
            paths.extend(self.background_dir.rglob(ext))
        return sorted(paths)

    def load_random_background(self) -> np.ndarray:
        """Loads a random background image and scales it."""
        bg_path = random.choice(self.background_paths)
        bg = cv2.imread(str(bg_path))

        if bg is None:
            # Fallback: Solid color background
            print(f"Warning: Could not load {bg_path}, using gray background")
            bg = np.full((*self.output_size[::-1], 3), 128, dtype=np.uint8)
            return bg

        # Scale to target size (with crop if necessary)
        bg = self._resize_and_crop(bg, self.output_size)

        return bg

    def _resize_and_crop(
        self,
        img: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Scales and crops an image to the target size.
        Maintains aspect ratio and crops centrally.
        """
        target_w, target_h = target_size
        h, w = img.shape[:2]

        # Calculate scaling factor (so image is at least target size)
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Scale
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop
        start_x = (new_w - target_w) // 2
        start_y = (new_h - target_h) // 2
        img = img[start_y:start_y + target_h, start_x:start_x + target_w]

        return img

    def composite(
        self,
        foreground: np.ndarray,
        background: np.ndarray,
        offset: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Combines foreground (with alpha) and background.

        Args:
            foreground: BGRA image (4 channels)
            background: BGR image (3 channels)
            offset: (x, y) offset of the foreground

        Returns:
            Combined BGR image
        """
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Apply offset
        ox, oy = offset

        # Result image (copy of background)
        result = background.copy()

        # Calculate area where foreground is placed
        # Foreground area
        fg_x1 = max(0, -ox)
        fg_y1 = max(0, -oy)
        fg_x2 = min(fg_w, bg_w - ox)
        fg_y2 = min(fg_h, bg_h - oy)

        # Background area
        bg_x1 = max(0, ox)
        bg_y1 = max(0, oy)
        bg_x2 = bg_x1 + (fg_x2 - fg_x1)
        bg_y2 = bg_y1 + (fg_y2 - fg_y1)

        if fg_x2 <= fg_x1 or fg_y2 <= fg_y1:
            # No overlap area
            return result

        # Extract alpha channel and normalize
        alpha = foreground[fg_y1:fg_y2, fg_x1:fg_x2, 3:4] / 255.0

        # Foreground RGB
        fg_rgb = foreground[fg_y1:fg_y2, fg_x1:fg_x2, :3]

        # Background area
        bg_region = result[bg_y1:bg_y2, bg_x1:bg_x2]

        # Alpha blending
        blended = (fg_rgb * alpha + bg_region * (1 - alpha)).astype(np.uint8)
        result[bg_y1:bg_y2, bg_x1:bg_x2] = blended

        return result

    def augment_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Augments a single image with a random background.

        Args:
            image_path: Path to PNG with transparency
            output_path: Optional output path

        Returns:
            Augmented image (BGR)
        """
        # Load foreground (with alpha)
        fg = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if fg is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Check if alpha channel exists
        if fg.shape[2] != 4:
            print(f"Warning: {image_path} has no alpha channel, skipping")
            return cv2.imread(str(image_path))

        # Scale to target size if necessary
        if fg.shape[:2] != self.output_size[::-1]:
            fg = cv2.resize(fg, self.output_size, interpolation=cv2.INTER_LINEAR)

        # Load random background
        bg = self.load_random_background()

        # Calculate offset (if enabled)
        offset = (0, 0)
        if self.offset_enabled:
            max_offset_x = int(self.output_size[0] * self.max_offset_fraction)
            max_offset_y = int(self.output_size[1] * self.max_offset_fraction)
            offset = (
                random.randint(-max_offset_x, max_offset_x),
                random.randint(-max_offset_y, max_offset_y)
            )

        # Compositing
        result = self.composite(fg, bg, offset)

        # Save if desired
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result)

        return result

    def augment_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        num_variations: int = 1,
        copy_labels: bool = True,
        labels_dir: Optional[Path] = None
    ) -> dict:
        """
        Augments an entire dataset.

        Args:
            input_dir: Folder with original images (PNG with transparency)
            output_dir: Output folder
            num_variations: Number of variations per image
            copy_labels: Copy labels
            labels_dir: Folder with labels (default: input_dir/../labels)

        Returns:
            Statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        # Labels folder
        if labels_dir is None:
            labels_dir = input_dir.parent / 'labels'

        # Create output folder
        output_images = output_dir / 'images'
        output_labels = output_dir / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        if copy_labels:
            output_labels.mkdir(parents=True, exist_ok=True)

        # Find all PNG files
        image_paths = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.PNG')))
        print(f"Found: {len(image_paths)} images")

        stats = {
            'input_images': len(image_paths),
            'output_images': 0,
            'skipped': 0
        }

        for img_path in tqdm(image_paths, desc="Augmenting"):
            stem = img_path.stem

            for var_idx in range(num_variations):
                # Output name
                if num_variations > 1:
                    out_name = f"{stem}_var{var_idx:03d}.png"
                else:
                    out_name = f"{stem}.png"

                out_path = output_images / out_name

                try:
                    # Augment
                    self.augment_image(img_path, out_path)
                    stats['output_images'] += 1

                    # Copy label
                    if copy_labels:
                        # JSON Label
                        json_label = labels_dir / f"{stem}.json"
                        if json_label.exists():
                            out_label_name = out_name.replace('.png', '.json').replace('.PNG', '.json')
                            shutil.copy2(json_label, output_labels / out_label_name)

                        # TXT Label (YOLO Format)
                        txt_label = labels_dir / f"{stem}.txt"
                        if txt_label.exists():
                            out_label_name = out_name.replace('.png', '.txt').replace('.PNG', '.txt')
                            shutil.copy2(txt_label, output_labels / out_label_name)

                except Exception as e:
                    print(f"Error at {img_path}: {e}")
                    stats['skipped'] += 1

        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Background Augmentation for Dartboard Images'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Folder with original images (PNG with transparency)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output folder'
    )
    parser.add_argument(
        '--backgrounds', '-b',
        type=str,
        required=True,
        help='Folder with background images'
    )
    parser.add_argument(
        '--variations', '-v',
        type=int,
        default=1,
        help='Number of variations per image (default: 1)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=800,
        help='Output size in pixels (default: 800)'
    )
    parser.add_argument(
        '--offset',
        action='store_true',
        help='Randomly shift dartboard (disabled by default)'
    )
    parser.add_argument(
        '--max-offset',
        type=float,
        default=0.1,
        help='Maximum offset as a fraction (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random Seed (default: 42)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Labels folder (default: input/../labels)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Do not copy labels'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Background Augmentation")
    print("=" * 60)
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Backgrounds:  {args.backgrounds}")
    print(f"Variations:   {args.variations}")
    print(f"Size:         {args.size}x{args.size}")
    print(f"Offset:       {'Enabled' if args.offset else 'Disabled'}")
    print("=" * 60)

    # Create augmentor
    augmentor = BackgroundAugmentor(
        background_dir=Path(args.backgrounds),
        output_size=(args.size, args.size),
        offset_enabled=args.offset,
        max_offset_fraction=args.max_offset,
        seed=args.seed
    )

    # Augment dataset
    stats = augmentor.augment_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        num_variations=args.variations,
        copy_labels=not args.no_labels,
        labels_dir=Path(args.labels) if args.labels else None
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"Input Images:  {stats['input_images']}")
    print(f"Output Images: {stats['output_images']}")
    print(f"Skipped:       {stats['skipped']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
