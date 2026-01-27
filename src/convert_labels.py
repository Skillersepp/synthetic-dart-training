"""
Label Converter: JSON -> YOLO Format

Konvertiert die synthetischen Dartboard-Labels vom JSON-Format
in das YOLO-Format für Object Detection.

Features:
- Jeder Keypoint wird zu einer kleinen Bounding Box
- Klassen: 0=Dart, 1=Center, 2-5=Kalibrationspunkte K1-K4
- On-The-Fly Background Augmentation (optional)
- Mehrere Variationen pro Bild möglich
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

# BackgroundProvider importieren falls vorhanden
try:
    from dataset import BackgroundProvider
    BG_PROVIDER_AVAILABLE = True
except ImportError:
    BG_PROVIDER_AVAILABLE = False


# Klassen-Mapping
CLASS_MAPPING = {
    'dart': 0,
    'Dartboard_Center': 1,
    'Dartboard_K1': 2,
    'Dartboard_K2': 3,
    'Dartboard_K3': 4,
    'Dartboard_K4': 5,
}


def load_json_label(json_path: Path) -> Dict:
    """Lädt eine JSON Label-Datei."""
    with open(json_path, 'r') as f:
        return json.load(f)


def keypoint_to_bbox(x: float, y: float, bbox_size: float) -> Tuple[float, float, float, float]:
    """
    Konvertiert einen Keypoint zu einer Bounding Box.

    Args:
        x, y: Normalisierte Keypoint-Koordinaten (0-1)
        bbox_size: Größe der BBox als Fraction der Bildgröße

    Returns:
        (x_center, y_center, width, height) - alle normalisiert
    """
    # Der Keypoint ist bereits das Zentrum
    return (x, y, bbox_size, bbox_size)


def convert_single_label(
    label_data: Dict,
    bbox_size: float = 0.025
) -> List[str]:
    """
    Konvertiert ein einzelnes JSON-Label zu YOLO-Format.

    Args:
        label_data: Geladene JSON-Daten
        bbox_size: Größe der Keypoint-BBoxen

    Returns:
        Liste von YOLO-Format Strings (eine Zeile pro Objekt)
    """
    lines = []

    # 1. Dartboard Keypoints (Kalibrierung)
    if 'dartboard' in label_data and 'keypoints' in label_data['dartboard']:
        for kp in label_data['dartboard']['keypoints']:
            name = kp['name']
            x, y = kp['x'], kp['y']
            is_visible = kp.get('is_visible', True)

            if not is_visible:
                continue

            # Prüfen ob Keypoint im gültigen Bereich liegt
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

            # Prüfen ob Dart im gültigen Bereich liegt
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
    """Erstellt die dataset.yaml für YOLO."""
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

    print(f"Dataset YAML erstellt: {yaml_path}")


def apply_background(
    image_path: Path,
    bg_provider,
    output_size: Tuple[int, int] = (800, 800)
) -> np.ndarray:
    """
    Lädt ein PNG mit Transparenz und fügt einen Hintergrund hinzu.

    Args:
        image_path: Pfad zum PNG-Bild
        bg_provider: BackgroundProvider Instanz
        output_size: Ausgabegröße (width, height)

    Returns:
        BGR Bild ohne Transparenz
    """
    # Bild mit Alpha laden
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Konnte Bild nicht laden: {image_path}")

    # Auf Zielgröße skalieren
    if img.shape[:2] != output_size[::-1]:
        img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

    # Falls kein Alpha-Kanal, direkt zurückgeben
    if img.shape[2] != 4:
        return img[:, :, :3]

    # Hintergrund holen
    bg = bg_provider.get_random()

    # Alpha-Blending
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
    output_size: int = 800
) -> Dict[str, int]:
    """
    Konvertiert den gesamten Datensatz von JSON zu YOLO-Format.

    Args:
        input_dir: Pfad zum Eingabe-Datensatz (mit images/ und labels/)
        output_dir: Pfad zum Ausgabe-Datensatz
        bbox_size: Größe der Keypoint-BBoxen
        train_ratio, val_ratio, test_ratio: Aufteilung der Daten
        seed: Random Seed für reproduzierbare Splits
        background_source: Pfad zu Hintergrundbildern oder None
        num_variations: Anzahl Variationen pro Bild (mit unterschiedl. Hintergründen)
        output_size: Ausgabe-Bildgröße

    Returns:
        Dictionary mit Statistiken
    """
    random.seed(seed)
    np.random.seed(seed)

    # Background Provider erstellen falls gewünscht
    bg_provider = None
    if background_source and BG_PROVIDER_AVAILABLE:
        from dataset import BackgroundProvider
        bg_provider = BackgroundProvider(
            source=background_source,
            output_size=(output_size, output_size)
        )
        print(f"Background Augmentation aktiviert: {num_variations} Variationen pro Bild")
    elif background_source and not BG_PROVIDER_AVAILABLE:
        print("WARNUNG: BackgroundProvider nicht verfügbar, überspringe Augmentation")

    # Pfade
    input_images = input_dir / 'images'
    input_labels = input_dir / 'labels'

    # Prüfen ob Input existiert
    if not input_images.exists():
        raise FileNotFoundError(f"Images Ordner nicht gefunden: {input_images}")
    if not input_labels.exists():
        raise FileNotFoundError(f"Labels Ordner nicht gefunden: {input_labels}")

    # Output-Struktur erstellen
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Alle JSON-Dateien sammeln
    json_files = sorted(list(input_labels.glob('*.json')))
    print(f"Gefunden: {len(json_files)} Label-Dateien")

    if len(json_files) == 0:
        raise ValueError("Keine JSON-Dateien gefunden!")

    # Shuffle und Split
    random.shuffle(json_files)
    n_total = len(json_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    split_indices = {
        'train': json_files[:n_train],
        'val': json_files[n_train:n_train + n_val],
        'test': json_files[n_train + n_val:]
    }

    # Statistiken
    stats = {
        'total': n_total,
        'train': len(split_indices['train']),
        'val': len(split_indices['val']),
        'test': len(split_indices['test']),
        'total_darts': 0,
        'total_keypoints': 0,
        'skipped': 0
    }

    # Konvertieren
    for split, files in split_indices.items():
        print(f"\nKonvertiere {split} Split ({len(files)} Dateien)...")

        for json_path in tqdm(files, desc=split):
            # Bildname ableiten
            stem = json_path.stem

            # Mögliche Bild-Extensions
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                candidate = input_images / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                print(f"Warnung: Kein Bild gefunden für {stem}")
                stats['skipped'] += 1
                continue

            # Label konvertieren
            try:
                label_data = load_json_label(json_path)
                yolo_lines = convert_single_label(label_data, bbox_size)
            except Exception as e:
                print(f"Fehler bei {json_path}: {e}")
                stats['skipped'] += 1
                continue

            if len(yolo_lines) == 0:
                print(f"Warnung: Keine gültigen Labels für {stem}")
                stats['skipped'] += 1
                continue

            # Statistiken aktualisieren
            n_darts = sum(1 for line in yolo_lines if line.startswith('0 '))
            n_keypoints = len(yolo_lines) - n_darts

            # Anzahl Variationen (nur für Training, Val/Test bekommen 1)
            n_vars = num_variations if split == 'train' and bg_provider else 1

            for var_idx in range(n_vars):
                # Dateiname mit Variation
                if n_vars > 1:
                    out_stem = f"{stem}_v{var_idx:03d}"
                else:
                    out_stem = stem

                # Bild verarbeiten
                if bg_provider:
                    # Mit Hintergrund-Augmentation
                    try:
                        augmented_img = apply_background(
                            img_path, bg_provider, (output_size, output_size)
                        )
                        dst_img = output_dir / 'images' / split / f"{out_stem}.png"
                        cv2.imwrite(str(dst_img), augmented_img)
                    except Exception as e:
                        print(f"Fehler bei Augmentation {img_path}: {e}")
                        continue
                else:
                    # Ohne Augmentation: Bild kopieren
                    dst_img = output_dir / 'images' / split / f"{out_stem}{img_path.suffix}"
                    shutil.copy2(img_path, dst_img)

                # Label schreiben
                dst_label = output_dir / 'labels' / split / f"{out_stem}.txt"
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                stats['total_darts'] += n_darts
                stats['total_keypoints'] += n_keypoints

    # Dataset YAML erstellen
    create_dataset_yaml(output_dir)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Konvertiert JSON-Labels zu YOLO-Format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='datasets/dataset_0',
        help='Eingabe-Datensatz Pfad'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='datasets/dataset_0_yolo',
        help='Ausgabe-Datensatz Pfad'
    )
    parser.add_argument(
        '--bbox-size',
        type=float,
        default=0.025,
        help='Keypoint BBox Größe (Fraction, default: 0.025 = 2.5%%)'
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
        help='Hintergrund-Ordner oder "textures" für auto-download'
    )
    parser.add_argument(
        '--variations', '-v',
        type=int,
        default=1,
        help='Anzahl Hintergrund-Variationen pro Bild (default: 1)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=800,
        help='Ausgabe-Bildgröße (default: 800)'
    )

    args = parser.parse_args()

    # Pfade relativ zum Script-Verzeichnis
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / args.input
    output_dir = script_dir / args.output
    bg_source = args.backgrounds
    if bg_source and not bg_source.startswith('/') and not bg_source.startswith('C:') and bg_source not in ['textures', 'coco', 'imagenet']:
        bg_source = str(script_dir / bg_source)

    print("=" * 50)
    print("JSON to YOLO Label Converter")
    print("=" * 50)
    print(f"Input:       {input_dir}")
    print(f"Output:      {output_dir}")
    print(f"BBox Size:   {args.bbox_size} ({args.bbox_size * 100:.1f}%)")
    print(f"Split:       {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"Backgrounds: {args.backgrounds or 'None'}")
    print(f"Variations:  {args.variations}")
    print(f"Image Size:  {args.size}")
    print("=" * 50)

    # Konvertieren
    stats = convert_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        bbox_size=args.bbox_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        background_source=bg_source,
        num_variations=args.variations,
        output_size=args.size
    )

    # Statistiken ausgeben
    print("\n" + "=" * 50)
    print("Konvertierung abgeschlossen!")
    print("=" * 50)
    print(f"Gesamt Bilder:     {stats['total']}")
    print(f"  - Train:         {stats['train']}")
    print(f"  - Validation:    {stats['val']}")
    print(f"  - Test:          {stats['test']}")
    print(f"  - Übersprungen:  {stats['skipped']}")
    print(f"Gesamt Darts:      {stats['total_darts']}")
    print(f"Gesamt Keypoints:  {stats['total_keypoints']}")
    print(f"\nDataset YAML: {output_dir / 'dataset.yaml'}")


if __name__ == '__main__':
    main()
