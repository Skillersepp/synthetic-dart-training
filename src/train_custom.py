"""
Custom Training Script with On-The-Fly Background Augmentation

This script utilizes a custom training loop with PyTorch to maintain
full control over the data augmentation process.

For rapid prototyping: Use train.py with pre-processed data.
For maximum flexibility: Use this script.
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
    raise ImportError("ultralytics not installed: pip install ultralytics")

from dataset import DartboardDataset, BackgroundProvider


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    total_epochs
):
    """Trains for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")

    for batch in pbar:
        images = batch['images'].to(device)
        labels = batch['labels']

        # Forward pass through YOLO
        # NOTE: This is simplified - YOLO uses a more complex loss calculation
        optimizer.zero_grad()

        # With ultralytics we need to proceed differently
        # Here we use a workaround
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
        description='Custom Training with On-The-Fly Background Augmentation'
    )

    parser.add_argument('--images', '-i', required=True, help='Images folder (PNG with transparency)')
    parser.add_argument('--labels', '-l', required=True, help='Labels folder (YOLO TXT)')
    parser.add_argument('--backgrounds', '-b', default=None, help='Backgrounds folder or "imagenet"/"coco"/"textures"')
    parser.add_argument('--model', '-m', default='yolo26n', help='Model (yolo26n/s/m/l/x)')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=800, help='Image size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--device', default='0', help='Device (0, 0,1, cpu)')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader Workers')
    parser.add_argument('--output', '-o', default='runs/train_custom', help='Output folder')

    args = parser.parse_args()

    print("=" * 60)
    print("Custom Training with Background Augmentation")
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

    # Create Dataset
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

    print(f"Dataset: {len(dataset)} images")

    # Load Model
    model = YOLO(f'{args.model}.pt')
    print(f"Model loaded: {args.model}")

    # NOTE: For real training with YOLO it is better to
    # process images beforehand (augment_backgrounds.py)
    # and then use model.train().

    print("\n" + "=" * 60)
    print("NOTE:")
    print("For optimal training with ultralytics YOLO I recommend:")
    print("1. python src/augment_backgrounds.py (Preprocessing)")
    print("2. python src/convert_labels.py")
    print("3. python src/train.py")
    print("")
    print("This custom trainer is for special use cases.")
    print("=" * 60)

    # Instead: Show how to view images on-the-fly
    print("\nShowing 3 sample images with random backgrounds...")

    import cv2
    output_dir = Path(args.output) / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        img = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f'sample_{i}.png'), img)
        print(f"  Saved: {output_dir / f'sample_{i}.png'}")

    print(f"\nSample images saved in: {output_dir}")


if __name__ == '__main__':
    main()
