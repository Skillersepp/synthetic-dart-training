"""
YOLO26 Training Script für Dartboard Detection

Trainiert ein YOLO26 Modell zur Erkennung von:
- Dart-Positionen (Keypoints)
- Kalibrationspunkten (Center + K1-K4)
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics nicht installiert. Bitte installieren mit:\n"
        "pip install ultralytics"
    )


def load_config(config_path: Path) -> dict:
    """Lädt die Training-Konfiguration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_name(base_name: str) -> str:
    """Generiert YOLO26 Modellnamen."""
    # Mapping von Kurzform zu vollständigem Namen
    model_map = {
        'yolo26n': 'yolo26n.pt',
        'yolo26s': 'yolo26s.pt',
        'yolo26m': 'yolo26m.pt',
        'yolo26l': 'yolo26l.pt',
        'yolo26x': 'yolo26x.pt',
    }
    return model_map.get(base_name, f'{base_name}.pt')


def train(
    dataset_yaml: str,
    config_path: str = None,
    model_size: str = 'yolo26n',
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 800,
    device: str = None,
    project: str = 'runs/train',
    name: str = None,
    resume: bool = False,
    pretrained: bool = True,
    patience: int = 20,
    workers: int = 8,
    **kwargs
):
    """
    Trainiert ein YOLO26 Modell.

    Args:
        dataset_yaml: Pfad zur Dataset YAML
        config_path: Optional Pfad zur Training-Config
        model_size: Modellgröße (yolo26n, yolo26s, yolo26m, yolo26l, yolo26x)
        epochs: Anzahl Epochen
        batch_size: Batch-Größe
        imgsz: Input-Bildgröße
        device: Device (None=auto, '0', '0,1', 'cpu')
        project: Ausgabe-Ordner
        name: Experiment-Name
        resume: Training fortsetzen
        pretrained: Pretrained Weights verwenden
        patience: Early Stopping Patience
        workers: Anzahl Dataloader Workers
        **kwargs: Weitere YOLO Training-Argumente
    """
    # Config laden falls vorhanden
    config = {}
    if config_path and Path(config_path).exists():
        config = load_config(Path(config_path))
        print(f"Config geladen: {config_path}")

    # Parameter aus Config übernehmen (falls nicht explizit gesetzt)
    training_cfg = config.get('training', {})
    aug_cfg = config.get('augmentation', {})

    epochs = training_cfg.get('epochs', epochs)
    batch_size = training_cfg.get('batch_size', batch_size)
    imgsz = training_cfg.get('imgsz', imgsz)
    patience = training_cfg.get('patience', patience)

    # Experiment-Name
    if name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f'dartboard_{model_size}_{timestamp}'

    # Modell laden
    model_name = get_model_name(model_size)
    print(f"\n{'='*60}")
    print(f"YOLO26 Dartboard Training")
    print(f"{'='*60}")
    print(f"Modell: {model_name}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {imgsz}")
    print(f"Device: {device or 'auto'}")
    print(f"Output: {project}/{name}")
    print(f"{'='*60}\n")

    # YOLO Modell initialisieren
    if resume:
        # Training fortsetzen
        model = YOLO(f'{project}/{name}/weights/last.pt')
    elif pretrained:
        # Mit pretrained Weights starten
        model = YOLO(model_name)
    else:
        # Von Scratch
        model = YOLO(model_name.replace('.pt', '.yaml'))

    # Augmentation-Parameter
    augment_args = {
        'degrees': aug_cfg.get('degrees', 180),
        'translate': aug_cfg.get('translate', 0.1),
        'scale': aug_cfg.get('scale', 0.3),
        'shear': aug_cfg.get('shear', 0.0),
        'perspective': aug_cfg.get('perspective', 0.0002),
        'flipud': aug_cfg.get('flipud', 0.5),
        'fliplr': aug_cfg.get('fliplr', 0.5),
        'hsv_h': aug_cfg.get('hsv_h', 0.015),
        'hsv_s': aug_cfg.get('hsv_s', 0.5),
        'hsv_v': aug_cfg.get('hsv_v', 0.3),
        'mosaic': aug_cfg.get('mosaic', 0.0),
        'mixup': aug_cfg.get('mixup', 0.0),
        'copy_paste': aug_cfg.get('copy_paste', 0.0),
    }

    # Training starten
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        resume=resume,
        pretrained=pretrained,
        # Optimizer
        optimizer=training_cfg.get('optimizer', 'AdamW'),
        lr0=training_cfg.get('lr0', 0.001),
        lrf=training_cfg.get('lrf', 0.01),
        momentum=training_cfg.get('momentum', 0.937),
        weight_decay=training_cfg.get('weight_decay', 0.0005),
        warmup_epochs=training_cfg.get('warmup_epochs', 3),
        warmup_momentum=training_cfg.get('warmup_momentum', 0.8),
        warmup_bias_lr=training_cfg.get('warmup_bias_lr', 0.1),
        # Augmentation
        **augment_args,
        # Loss
        box=config.get('loss', {}).get('box', 7.5),
        cls=config.get('loss', {}).get('cls', 0.5),
        dfl=config.get('loss', {}).get('dfl', 1.5),
        # Zusätzliche Argumente
        plots=True,
        save=True,
        save_period=config.get('output', {}).get('save_period', 10),
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
        description='YOLO26 Training für Dartboard Detection'
    )

    # Erforderliche Argumente
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='datasets/dataset_0_yolo/dataset.yaml',
        help='Pfad zur Dataset YAML'
    )

    # Optionale Argumente
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/train_config.yaml',
        help='Pfad zur Training-Config'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolo26n',
        choices=['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x'],
        help='Modellgröße'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Anzahl Epochen'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch-Größe'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=800,
        help='Input-Bildgröße'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (None=auto, 0, 0,1, cpu)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Ausgabe-Ordner'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment-Name'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Training fortsetzen'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Ohne pretrained Weights'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early Stopping Patience'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Dataloader Workers'
    )

    args = parser.parse_args()

    # Pfade relativ zum Script-Verzeichnis auflösen
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / args.data
    config_path = script_dir / args.config if args.config else None

    # Training starten
    train(
        dataset_yaml=str(data_path),
        config_path=str(config_path) if config_path else None,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(script_dir / args.project),
        name=args.name,
        resume=args.resume,
        pretrained=not args.no_pretrained,
        patience=args.patience,
        workers=args.workers
    )


if __name__ == '__main__':
    main()
