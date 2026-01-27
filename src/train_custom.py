"""
Custom Training Script mit On-The-Fly Background Augmentation

Dieses Script nutzt einen eigenen Training-Loop mit PyTorch,
um volle Kontrolle über die Augmentation zu haben.

Für schnelles Prototyping: Nutze train.py mit vorverarbeiteten Daten.
Für maximale Flexibilität: Nutze dieses Script.
"""

import argparse
from pathlib import Path
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics nicht installiert: pip install ultralytics")

from dataset import DartboardDataset, BackgroundProvider


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    total_epochs
):
    """Trainiert eine Epoche."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")

    for batch in pbar:
        images = batch['images'].to(device)
        labels = batch['labels']

        # Forward pass durch YOLO
        # HINWEIS: Dies ist vereinfacht - YOLO hat komplexere Loss-Berechnung
        optimizer.zero_grad()

        # Bei ultralytics müssen wir anders vorgehen
        # Hier nutzen wir einen Workaround
        loss = model.model(images, labels)

        if isinstance(loss, dict):
            loss = sum(loss.values())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(
        description='Custom Training mit On-The-Fly Background Augmentation'
    )

    parser.add_argument('--images', '-i', required=True, help='Bilder-Ordner (PNG mit Transparenz)')
    parser.add_argument('--labels', '-l', required=True, help='Labels-Ordner (YOLO TXT)')
    parser.add_argument('--backgrounds', '-b', default=None, help='Hintergrund-Ordner oder "imagenet"/"coco"/"textures"')
    parser.add_argument('--model', '-m', default='yolo26n', help='Modell (yolo26n/s/m/l/x)')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Anzahl Epochen')
    parser.add_argument('--batch', type=int, default=16, help='Batch-Größe')
    parser.add_argument('--imgsz', type=int, default=800, help='Bildgröße')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--device', default='0', help='Device (0, 0,1, cpu)')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader Workers')
    parser.add_argument('--output', '-o', default='runs/train_custom', help='Output-Ordner')

    args = parser.parse_args()

    print("=" * 60)
    print("Custom Training mit Background Augmentation")
    print("=" * 60)
    print(f"Images:      {args.images}")
    print(f"Labels:      {args.labels}")
    print(f"Backgrounds: {args.backgrounds or 'Generated'}")
    print(f"Model:       {args.model}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch Size:  {args.batch}")
    print("=" * 60)

    # Device
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device.split(",")[0]}')

    print(f"Device: {device}")

    # Dataset erstellen
    dataset = DartboardDataset(
        images_dir=args.images,
        labels_dir=args.labels,
        background_source=args.backgrounds,
        img_size=args.imgsz,
        augment=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda batch: {
            'images': torch.stack([item['image'] for item in batch]),
            'labels': [item['labels'] for item in batch],
            'paths': [item['path'] for item in batch]
        }
    )

    print(f"Dataset: {len(dataset)} Bilder")

    # Model laden
    model = YOLO(f'{args.model}.pt')
    print(f"Model geladen: {args.model}")

    # HINWEIS: Für echtes Training mit YOLO ist es besser,
    # die Bilder vorher zu verarbeiten (augment_backgrounds.py)
    # und dann model.train() zu nutzen.

    print("\n" + "=" * 60)
    print("HINWEIS:")
    print("Für optimales Training mit ultralytics YOLO empfehle ich:")
    print("1. python src/augment_backgrounds.py (Preprocessing)")
    print("2. python src/convert_labels.py")
    print("3. python src/train.py")
    print("")
    print("Dieser Custom-Trainer ist für spezielle Anwendungsfälle.")
    print("=" * 60)

    # Stattdessen: Zeige wie man die Bilder on-the-fly sehen kann
    print("\nZeige 3 Beispielbilder mit zufälligen Hintergründen...")

    import cv2
    output_dir = Path(args.output) / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        img = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f'sample_{i}.png'), img)
        print(f"  Gespeichert: {output_dir / f'sample_{i}.png'}")

    print(f"\nBeispielbilder gespeichert in: {output_dir}")


if __name__ == '__main__':
    main()
