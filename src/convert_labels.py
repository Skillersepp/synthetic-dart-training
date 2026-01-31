"""
Label Converter: JSON -> YOLO Format

Converts the synthetic dartboard labels from JSON format
into the YOLO format for object detection.

Features:
- Each keypoint becomes a small bounding box
- Classes: 0=Dart, 1=Center, 2-5=Calibration points K1-K4
- On-The-Fly Background Augmentation (optional)
- Multiple variations per image supported
"""

import json
import os
import shutil
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml
import cv2
import numpy as np

# Import BackgroundProvider if available
try:
    from dataset import BackgroundProvider
    BG_PROVIDER_AVAILABLE = True
except ImportError:
    BG_PROVIDER_AVAILABLE = False


# Class Mapping
# Automatisch erweitert für beliebig viele Kalibrationspunkte
CLASS_MAPPING = {
    'dart': 0,
    'Dartboard_Center': 1,
    'Dartboard_K1': 2,
    'Dartboard_K2': 3,
    'Dartboard_K3': 4,
    'Dartboard_K4': 5,
    # Neue Kalibrationspunkte auf Triple Ring (äußere Kante)
    'Dartboard_KT1': 6,  # Triple Ring oben (Richtung 20)
    'Dartboard_KT2': 7,  # Triple Ring unten (Richtung 3)
    'Dartboard_KT3': 8,  # Triple Ring links (Richtung 6)
    'Dartboard_KT4': 9,  # Triple Ring rechts (Richtung 11)
}


def load_json_label(json_path: Path) -> Dict:
    """Loads a JSON label file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def discover_keypoint_classes(input_labels_dir: Path) -> Dict[str, int]:
    """
    Automatisch alle Keypoint-Klassen aus dem Datensatz extrahieren.

    Scannt die ersten N JSON-Dateien und sammelt alle einzigartigen Keypoint-Namen.
    Nützlich, wenn neue Kalibrationspunkte hinzugefügt werden.

    Args:
        input_labels_dir: Pfad zum Labels-Ordner

    Returns:
        Erweitertes CLASS_MAPPING mit allen gefundenen Keypoints
    """
    discovered_keypoints = set()

    # Scanne erste 10 Dateien
    json_files = sorted(list(input_labels_dir.glob('*.json')))[:10]

    for json_path in json_files:
        try:
            data = load_json_label(json_path)
            if 'dartboard' in data and 'keypoints' in data['dartboard']:
                for kp in data['dartboard']['keypoints']:
                    discovered_keypoints.add(kp['name'])
        except Exception:
            continue

    # Erstelle erweitertes Mapping
    extended_mapping = {'dart': 0}

    # Sortiere Keypoints: Center zuerst, dann K-Punkte, dann KT-Punkte, dann Rest
    sorted_keypoints = sorted(discovered_keypoints, key=lambda x: (
        0 if 'Center' in x else
        1 if x.startswith('Dartboard_K') and not x.startswith('Dartboard_KT') else
        2 if x.startswith('Dartboard_KT') else
        3,
        x
    ))

    for idx, kp_name in enumerate(sorted_keypoints, start=1):
        extended_mapping[kp_name] = idx

    return extended_mapping


def keypoint_to_bbox(x: float, y: float, bbox_size: float) -> Tuple[float, float, float, float]:
    """
    Converts a keypoint to a bounding box.

    Args:
        x, y: Normalized keypoint coordinates (0-1)
        bbox_size: Size of the BBox as a fraction of the image size

    Returns:
        (x_center, y_center, width, height) - all normalized
    """
    # The keypoint is already the center
    return (x, y, bbox_size, bbox_size)


def convert_single_label(
    label_data: Dict,
    bbox_size: float = 0.025
) -> List[str]:
    """
    Converts a single JSON label to YOLO format.

    Args:
        label_data: Loaded JSON data
        bbox_size: Size of the keypoint BBoxes

    Returns:
        List of YOLO format strings (one line per object)
    """
    lines = []

    # 1. Dartboard Keypoints (Calibration)
    if 'dartboard' in label_data and 'keypoints' in label_data['dartboard']:
        for kp in label_data['dartboard']['keypoints']:
            name = kp['name']
            x, y = kp['x'], kp['y']
            is_visible = kp.get('is_visible', True)

            if not is_visible:
                continue

            # Check if keypoint is within valid range
            half_box = bbox_size / 2
            if x - half_box < 0 or x + half_box > 1 or y - half_box < 0 or y + half_box > 1:
                continue

            if name in CLASS_MAPPING:
                class_id = CLASS_MAPPING[name]
                x_center, y_center, w, h = keypoint_to_bbox(x, y, bbox_size)
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # 2. Darts
    if 'darts' in label_data:
        for dart in label_data['darts']:
            x, y = dart['x'], dart['y']
            is_visible = dart.get('is_visible', True)

            if not is_visible:
                continue

            # Check if dart is within valid range
            half_box = bbox_size / 2
            if x - half_box < 0 or x + half_box > 1 or y - half_box < 0 or y + half_box > 1:
                continue

            class_id = CLASS_MAPPING['dart']
            x_center, y_center, w, h = keypoint_to_bbox(x, y, bbox_size)
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return lines


def create_dataset_yaml(
    output_dir: Path,
    train_dir: str = "images/train",
    val_dir: str = "images/val",
    test_dir: str = "images/test"
) -> None:
    """Creates the dataset.yaml for YOLO."""
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
        'nc': len(CLASS_MAPPING),
        'names': {v: k for k, v in CLASS_MAPPING.items()}
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset YAML created: {yaml_path}")


def apply_background(
    image_path: Path,
    bg_provider,
    output_size: Tuple[int, int] = (1024, 1024)
) -> np.ndarray:
    """
    Loads a PNG with transparency and adds a background.

    Args:
        image_path: Path to PNG image
        bg_provider: BackgroundProvider instance
        output_size: Output size (width, height)

    Returns:
        BGR image without transparency
    """
    # Load image with alpha
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Scale to target size
    if img.shape[:2] != output_size[::-1]:
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

    # If no alpha channel, return directly
    if img.shape[2] != 4:
        return img[:, :, :3]

    # Get background
    bg = bg_provider.get_random()

    # Alpha blending
    alpha = img[:, :, 3:4] / 255.0
    fg_rgb = img[:, :, :3]
    blended = (fg_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)

    return blended


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    bbox_size: float = 0.025,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    background_source: Optional[str] = None,
    num_variations: int = 1,
    output_size: int = 1024,
    auto_discover_classes: bool = True,
    move_files: bool = False
) -> Dict[str, int]:
    """
    Converts the entire dataset from JSON to YOLO format.

    Args:
        input_dir: Path to input dataset (with images/ and labels/)
        output_dir: Path to output dataset
        bbox_size: Size of keypoint BBoxes
        train_ratio, val_ratio, test_ratio: Data splits
        seed: Random seed for reproducible splits
        background_source: Path to background images or None (ignored if move_files=True)
        num_variations: Number of variations per image (ignored if move_files=True)
        output_size: Output image size (ignored if move_files=True)
        auto_discover_classes: Automatically detect all keypoint classes from dataset
        move_files: If True, move images instead of copy/transform (saves disk space)

    Returns:
        Dictionary with statistics
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create Background Provider if desired (but not in move mode)
    bg_provider = None
    if move_files:
        if background_source:
            print("WARNING: Background augmentation disabled in --move mode")
        if num_variations > 1:
            print("WARNING: Variations disabled in --move mode (num_variations ignored)")
    elif background_source and BG_PROVIDER_AVAILABLE:
        from dataset import BackgroundProvider
        bg_provider = BackgroundProvider(
            source=background_source,
            output_size=(output_size, output_size)
        )
        print(f"Background augmentation enabled: {num_variations} variations per image")
    elif background_source and not BG_PROVIDER_AVAILABLE:
        print("WARNING: BackgroundProvider not available, skipping augmentation")

    # Paths
    input_images = input_dir / 'images'
    input_labels = input_dir / 'labels'

    # Check if input exists
    if not input_images.exists():
        raise FileNotFoundError(f"Images folder not found: {input_images}")
    if not input_labels.exists():
        raise FileNotFoundError(f"Labels folder not found: {input_labels}")

    # Create output structure
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect all JSON files
    json_files = sorted(list(input_labels.glob('*.json')))
    print(f"Found: {len(json_files)} label files")

    if len(json_files) == 0:
        raise ValueError("No JSON files found!")

    # Auto-discover classes if enabled
    global CLASS_MAPPING
    if auto_discover_classes:
        discovered_mapping = discover_keypoint_classes(input_labels)
        if len(discovered_mapping) > len(CLASS_MAPPING):
            print(f"\nAuto-Discovery: Found {len(discovered_mapping)} classes (vs {len(CLASS_MAPPING)} predefined)")
            print("Discovered classes:")
            for name, class_id in sorted(discovered_mapping.items(), key=lambda x: x[1]):
                status = "✓" if name in CLASS_MAPPING else "NEW"
                print(f"  [{class_id}] {name:25s} {status}")
            CLASS_MAPPING = discovered_mapping
        else:
            print(f"Using predefined CLASS_MAPPING with {len(CLASS_MAPPING)} classes")

    # Shuffle and split
    random.shuffle(json_files)
    n_total = len(json_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    split_indices = {
        'train': json_files[:n_train],
        'val': json_files[n_train:n_train + n_val],
        'test': json_files[n_train + n_val:]
    }

    # Statistics
    stats = {
        'total': n_total,
        'train': len(split_indices['train']),
        'val': len(split_indices['val']),
        'test': len(split_indices['test']),
        'total_darts': 0,
        'total_keypoints': 0,
        'skipped': 0
    }

    # Convert
    for split, files in split_indices.items():
        print(f"\nConverting {split} split ({len(files)} files)...")

        for json_path in tqdm(files, desc=split):
            # Derive image name
            stem = json_path.stem

            # Possible image extensions
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                candidate = input_images / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"Warning: No image found for {stem}")
                stats['skipped'] += 1
                continue

            # Convert label
            try:
                label_data = load_json_label(json_path)
                yolo_lines = convert_single_label(label_data, bbox_size)
            except Exception as e:
                print(f"Error at {json_path}: {e}")
                stats['skipped'] += 1
                continue

            if len(yolo_lines) == 0:
                print(f"Warning: No valid labels for {stem}")
                stats['skipped'] += 1
                continue

            # Update statistics
            n_darts = sum(1 for line in yolo_lines if line.startswith('0 '))
            n_keypoints = len(yolo_lines) - n_darts

            # Number of variations (only for training, val/test get 1)
            # In move mode: always 1 (no variations)
            if move_files:
                n_vars = 1
            else:
                n_vars = num_variations if split == 'train' and bg_provider else 1

            for var_idx in range(n_vars):
                # Filename with variation
                if n_vars > 1:
                    out_stem = f"{stem}_v{var_idx:03d}"
                else:
                    out_stem = stem

                # Process image
                if move_files:
                    # Move mode: Just move the file without any transformation
                    dst_img = output_dir / 'images' / split / f"{out_stem}{img_path.suffix}"
                    shutil.move(str(img_path), str(dst_img))
                elif bg_provider:
                    # With background augmentation
                    try:
                        augmented_img = apply_background(
                            img_path, bg_provider, (output_size, output_size)
                        )
                        dst_img = output_dir / 'images' / split / f"{out_stem}.png"
                        cv2.imwrite(str(dst_img), augmented_img)
                    except Exception as e:
                        print(f"Error augmenting {img_path}: {e}")
                        continue
                else:
                    # Without augmentation: Copy image
                    dst_img = output_dir / 'images' / split / f"{out_stem}{img_path.suffix}"
                    shutil.copy2(img_path, dst_img)

                # Write label
                dst_label = output_dir / 'labels' / split / f"{out_stem}.txt"
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                stats['total_darts'] += n_darts
                stats['total_keypoints'] += n_keypoints

    # Create dataset YAML
    create_dataset_yaml(output_dir)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Converts JSON labels to YOLO format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='datasets/dataset_0',
        help='Input dataset path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='datasets/dataset_0_yolo',
        help='Output dataset path'
    )
    parser.add_argument(
        '--bbox-size',
        type=float,
        default=None,
        help='Keypoint BBox size (Fraction, default: from config or 0.025 = 2.5%%)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to training config (for bbox_size and other settings)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train Split Ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation Split Ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test Split Ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random Seed (default: 42)'
    )
    parser.add_argument(
        '--backgrounds', '-b',
        type=str,
        default=None,
        help='Background folder or "textures" for auto-download'
    )
    parser.add_argument(
        '--variations', '-v',
        type=int,
        default=1,
        help='Number of background variations per image (default: 1)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=1024,
        help='Output image size (default: 1024)'
    )
    parser.add_argument(
        '--no-auto-discover',
        action='store_true',
        help='Disable automatic class discovery (use predefined CLASS_MAPPING only)'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copy (saves disk space, but modifies source dataset!)'
    )

    args = parser.parse_args()

    # Paths relative to script directory
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / args.input
    output_dir = script_dir / args.output
    bg_source = args.backgrounds
    if bg_source and not bg_source.startswith('/') and not bg_source.startswith('C:') and bg_source not in ['textures', 'coco', 'imagenet']:
        bg_source = str(script_dir / bg_source)

    # Load config for defaults (if bbox_size not specified)
    config = {}
    if args.config:
        config_path = script_dir / args.config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

    # Get bbox_size from config if not specified
    bbox_size = args.bbox_size
    if bbox_size is None:
        bbox_size = config.get('dartboard', {}).get('keypoint_bbox_size', 0.025)

    print("=" * 50)
    print("JSON to YOLO Label Converter")
    print("=" * 50)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"BBox Size:   {bbox_size} ({bbox_size * 100:.1f}%)")
    print(f"Split:       {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"Mode:        {'MOVE (⚠️  source will be modified!)' if args.move else 'COPY'}")
    if not args.move:
        print(f"Backgrounds: {args.backgrounds or 'None'}")
        print(f"Variations:  {args.variations}")
        print(f"Image Size:  {args.size}")
    if args.config and config:
        print(f"Config:      {args.config} (loaded)")
    print("=" * 50)

    # Convert
    stats = convert_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        bbox_size=bbox_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        background_source=bg_source,
        num_variations=args.variations,
        output_size=args.size,
        auto_discover_classes=not args.no_auto_discover,
        move_files=args.move
    )

    # Print statistics
    print("\n" + "=" * 50)
    print("Conversion finished!")
    print("=" * 50)
    print(f"Total Images:      {stats['total']}")
    print(f"  - Train:         {stats['train']}")
    print(f"  - Validation:    {stats['val']}")
    print(f"  - Test:          {stats['test']}")
    print(f"  - Skipped:       {stats['skipped']}")
    print(f"Total Darts:       {stats['total_darts']}")
    print(f"Total Keypoints:   {stats['total_keypoints']}")
    print(f"\nDataset YAML: {output_dir / 'dataset.yaml'}")


if __name__ == '__main__':
    main()
