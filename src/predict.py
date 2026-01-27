"""
YOLO26 Prediction & Evaluation Script

Führt Inference durch und berechnet Metriken (PCS, MASE).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import cv2
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics nicht installiert. Bitte installieren mit:\n"
        "pip install ultralytics"
    )

from utils.scoring import DartScorer, calculate_pcs, calculate_mase
from utils.visualization import draw_predictions, save_prediction_image


# Klassen-Mapping (umgekehrt)
CLASS_NAMES = {
    0: 'dart',
    1: 'cal_center',
    2: 'cal_k1',
    3: 'cal_k2',
    4: 'cal_k3',
    5: 'cal_k4',
}


def parse_yolo_results(results) -> List[Dict]:
    """
    Parst YOLO Ergebnisse in ein einheitliches Format.

    Args:
        results: YOLO Prediction Results

    Returns:
        Liste von Detections mit class_id, x_center, y_center, width, height, confidence
    """
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            # Koordinaten (xyxy format zu xywh normalisiert)
            xyxy = boxes.xyxy[i].cpu().numpy()
            img_h, img_w = result.orig_shape

            x1, y1, x2, y2 = xyxy
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            detections.append({
                'class_id': int(boxes.cls[i].cpu().numpy()),
                'x_center': float(x_center),
                'y_center': float(y_center),
                'width': float(width),
                'height': float(height),
                'confidence': float(boxes.conf[i].cpu().numpy())
            })

    return detections


def detections_to_keypoints(detections: List[Dict]) -> Dict:
    """
    Konvertiert Detections zu Keypoint-Dictionary für Score-Berechnung.

    Args:
        detections: Liste von Detections

    Returns:
        Dictionary mit 'darts' und 'calibration' Listen
    """
    darts = []
    calibration = [None] * 5  # [center, k1, k2, k3, k4]

    for det in detections:
        class_id = det['class_id']
        point = (det['x_center'], det['y_center'])

        if class_id == 0:  # Dart
            darts.append(point)
        elif class_id == 1:  # Center
            calibration[0] = point
        elif 2 <= class_id <= 5:  # K1-K4
            calibration[class_id - 1] = point

    # Nur gültige Kalibrationspunkte behalten
    calibration = [p for p in calibration if p is not None]

    return {
        'darts': darts,
        'calibration': calibration
    }


def load_ground_truth(json_path: Path) -> Dict:
    """
    Lädt Ground Truth aus JSON-Datei.

    Returns:
        Dictionary mit 'darts' (Positionen + Scores) und 'calibration'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    darts = []
    dart_scores = []

    if 'darts' in data:
        for dart in data['darts']:
            darts.append((dart['x'], dart['y']))
            dart_scores.append(dart.get('score', 0))

    calibration = []
    if 'dartboard' in data and 'keypoints' in data['dartboard']:
        for kp in data['dartboard']['keypoints']:
            calibration.append((kp['x'], kp['y']))

    return {
        'darts': darts,
        'calibration': calibration,
        'dart_scores': dart_scores,
        'total_score': sum(dart_scores)
    }


def predict_single(
    model: YOLO,
    image_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7
) -> Tuple[List[Dict], np.ndarray]:
    """
    Führt Prediction für ein einzelnes Bild durch.

    Returns:
        (detections, image)
    """
    # Inference
    results = model.predict(
        str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    # Bild laden
    img = cv2.imread(str(image_path))

    # Ergebnisse parsen
    detections = parse_yolo_results(results)

    return detections, img


def evaluate(
    model_path: str,
    data_dir: str,
    split: str = 'test',
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    output_dir: str = None,
    write_images: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Evaluiert das Modell auf einem Datensatz.

    Args:
        model_path: Pfad zum trainierten Modell
        data_dir: Pfad zum YOLO-Datensatz
        split: 'train', 'val', oder 'test'
        conf_threshold: Confidence Threshold
        iou_threshold: IoU Threshold für NMS
        output_dir: Ausgabe-Ordner für Bilder
        write_images: Bilder mit Predictions speichern
        verbose: Detaillierte Ausgabe

    Returns:
        Dictionary mit Metriken
    """
    data_dir = Path(data_dir)
    model = YOLO(model_path)

    # Pfade
    images_dir = data_dir / 'images' / split
    labels_dir = data_dir / 'labels' / split

    # Suche nach Original-Labels (JSON)
    # Wir brauchen die JSON-Labels für die Ground Truth Scores
    original_labels_dir = data_dir.parent / data_dir.name.replace('_yolo', '') / 'labels'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images nicht gefunden: {images_dir}")

    # Output-Ordner
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Alle Bilder sammeln
    image_files = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    print(f"Evaluiere {len(image_files)} Bilder aus {split} Split...")

    # Scorer
    scorer = DartScorer()

    # Ergebnisse sammeln
    results = {
        'predictions': [],
        'ground_truth': [],
        'pred_scores': [],
        'gt_scores': [],
        'errors': []
    }

    for img_path in tqdm(image_files, desc=f"Evaluating {split}"):
        # Prediction
        detections, img = predict_single(
            model, img_path, conf_threshold, iou_threshold
        )

        # Keypoints extrahieren
        keypoints = detections_to_keypoints(detections)

        # Score berechnen (wenn genug Kalibrationspunkte)
        pred_score = 0
        score_strings = []

        if len(keypoints['calibration']) >= 4:
            try:
                has_center = len(keypoints['calibration']) >= 5
                scores = scorer.calculate_scores(
                    np.array(keypoints['darts']),
                    np.array(keypoints['calibration']),
                    has_center=has_center
                )
                pred_score = sum(s[1] for s in scores)
                score_strings = [s[0] for s in scores]
            except Exception as e:
                if verbose:
                    print(f"Score-Fehler bei {img_path.name}: {e}")

        # Ground Truth laden (falls vorhanden)
        gt_score = 0
        json_path = original_labels_dir / f"{img_path.stem}.json"
        if json_path.exists():
            gt_data = load_ground_truth(json_path)
            gt_score = gt_data['total_score']

        results['pred_scores'].append(pred_score)
        results['gt_scores'].append(gt_score)
        results['errors'].append(abs(pred_score - gt_score))

        # Bild speichern
        if write_images and output_dir:
            result_img = draw_predictions(img, detections, scores=score_strings)

            # Score-Info hinzufügen
            text = f"Pred: {pred_score} | GT: {gt_score}"
            cv2.putText(result_img, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imwrite(str(output_dir / img_path.name), result_img)

    # Metriken berechnen
    pcs = calculate_pcs(results['pred_scores'], results['gt_scores'])
    mase = calculate_mase(results['pred_scores'], results['gt_scores'])

    metrics = {
        'split': split,
        'num_images': len(image_files),
        'PCS': pcs,
        'MASE': mase,
        'errors': results['errors']
    }

    # Ausgabe
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({split})")
    print(f"{'='*50}")
    print(f"Bilder:                    {metrics['num_images']}")
    print(f"Percent Correct Score:     {pcs:.1f}%")
    print(f"Mean Absolute Score Error: {mase:.2f}")
    print(f"{'='*50}\n")

    return metrics


def predict_image(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.25,
    output_path: str = None,
    show: bool = False
) -> Dict:
    """
    Prediction für ein einzelnes Bild.

    Returns:
        Dictionary mit Detections und Scores
    """
    model = YOLO(model_path)
    image_path = Path(image_path)

    # Prediction
    detections, img = predict_single(model, image_path, conf_threshold)

    # Keypoints und Scores
    keypoints = detections_to_keypoints(detections)
    scorer = DartScorer()

    score_info = {'scores': [], 'total': 0}
    if len(keypoints['calibration']) >= 4:
        try:
            has_center = len(keypoints['calibration']) >= 5
            scores = scorer.calculate_scores(
                np.array(keypoints['darts']),
                np.array(keypoints['calibration']),
                has_center=has_center
            )
            score_info = {
                'scores': [(s[0], s[1]) for s in scores],
                'total': sum(s[1] for s in scores)
            }
        except Exception as e:
            print(f"Score-Berechnung fehlgeschlagen: {e}")

    # Visualisierung
    score_strings = [s[0] for s, _ in score_info.get('scores', [])] or None
    result_img = draw_predictions(img, detections, scores=score_strings)

    # Score-Text
    text = f"Total: {score_info['total']}"
    cv2.putText(result_img, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Speichern
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Gespeichert: {output_path}")

    # Anzeigen
    if show:
        cv2.imshow('Prediction', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        'detections': detections,
        'keypoints': keypoints,
        'scores': score_info
    }


def main():
    parser = argparse.ArgumentParser(
        description='YOLO26 Prediction & Evaluation'
    )

    subparsers = parser.add_subparsers(dest='command', help='Kommando')

    # Evaluate Subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Modell evaluieren')
    eval_parser.add_argument('--model', '-m', required=True, help='Modell-Pfad')
    eval_parser.add_argument('--data', '-d', required=True, help='Dataset-Pfad')
    eval_parser.add_argument('--split', '-s', default='test', help='Split')
    eval_parser.add_argument('--conf', type=float, default=0.25, help='Confidence')
    eval_parser.add_argument('--iou', type=float, default=0.7, help='IoU Threshold')
    eval_parser.add_argument('--output', '-o', help='Output-Ordner')
    eval_parser.add_argument('--write', action='store_true', help='Bilder speichern')

    # Predict Subcommand
    pred_parser = subparsers.add_parser('predict', help='Einzelbild Prediction')
    pred_parser.add_argument('--model', '-m', required=True, help='Modell-Pfad')
    pred_parser.add_argument('--image', '-i', required=True, help='Bild-Pfad')
    pred_parser.add_argument('--conf', type=float, default=0.25, help='Confidence')
    pred_parser.add_argument('--output', '-o', help='Output-Pfad')
    pred_parser.add_argument('--show', action='store_true', help='Bild anzeigen')

    args = parser.parse_args()

    if args.command == 'evaluate':
        evaluate(
            model_path=args.model,
            data_dir=args.data,
            split=args.split,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_dir=args.output,
            write_images=args.write
        )
    elif args.command == 'predict':
        result = predict_image(
            model_path=args.model,
            image_path=args.image,
            conf_threshold=args.conf,
            output_path=args.output,
            show=args.show
        )
        print(f"\nErgebnis:")
        print(f"  Darts gefunden: {len(result['keypoints']['darts'])}")
        print(f"  Kalibrationspunkte: {len(result['keypoints']['calibration'])}")
        print(f"  Scores: {result['scores']}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
