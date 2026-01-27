"""
YOLO26 Prediction & Evaluation Script

Runs inference and calculates metrics (PCS, MASE).
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
        "ultralytics not installed. Please install with:\n"
        "pip install ultralytics"
    )

from utils.scoring import DartScorer, calculate_pcs, calculate_mase
from utils.visualization import draw_predictions, save_prediction_image


# Class Mapping (reversed)
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
    Parses YOLO results into a unified format.

    Args:
        results: YOLO Prediction Results

    Returns:
        List of detections with class_id, x_center, y_center, width, height, confidence
    """
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            # Coordinates (xyxy format to xywh normalized)
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
    Converts detections to a keypoint dictionary for score calculation.

    Args:
        detections: List of detections

    Returns:
        Dictionary with 'darts' and 'calibration' lists
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

    # Only keep valid calibration points
    calibration = [p for p in calibration if p is not None]

    return {
        'darts': darts,
        'calibration': calibration
    }


def load_ground_truth(json_path: Path) -> Dict:
    """
    Loads Ground Truth from JSON file.

    Returns:
        Dictionary with 'darts' (positions + scores) and 'calibration'
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
    Performs prediction for a single image.

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

    # Load image
    img = cv2.imread(str(image_path))

    # Parse results
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
    Evaluates the model on a dataset.

    Args:
        model_path: Path to trained model
        data_dir: Path to YOLO dataset
        split: 'train', 'val', or 'test'
        conf_threshold: Confidence Threshold
        iou_threshold: IoU Threshold for NMS
        output_dir: Output folder for images
        write_images: Save images with predictions
        verbose: Detailed output

    Returns:
        Dictionary with metrics
    """
    data_dir = Path(data_dir)
    model = YOLO(model_path)

    # Paths
    images_dir = data_dir / 'images' / split
    labels_dir = data_dir / 'labels' / split

    # Search for original labels (JSON)
    # We need the JSON labels for the ground truth scores
    original_labels_dir = data_dir.parent / data_dir.name.replace('_yolo', '') / 'labels'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found: {images_dir}")

    # Output folder
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images
    image_files = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    print(f"Evaluating {len(image_files)} images from {split} split...")

    # Scorer
    scorer = DartScorer()

    # Collect results
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

        # Extract Keypoints
        keypoints = detections_to_keypoints(detections)

        # Calculate score (if enough calibration points)
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
                    print(f"Score error at {img_path.name}: {e}")

        # Load Ground Truth (if available)
        gt_score = 0
        json_path = original_labels_dir / f"{img_path.stem}.json"
        if json_path.exists():
            gt_data = load_ground_truth(json_path)
            gt_score = gt_data['total_score']

        results['pred_scores'].append(pred_score)
        results['gt_scores'].append(gt_score)
        results['errors'].append(abs(pred_score - gt_score))

        # Save image
        if write_images and output_dir:
            result_img = draw_predictions(img, detections, scores=score_strings)

            # Add Score Info
            text = f"Pred: {pred_score} | GT: {gt_score}"
            cv2.putText(result_img, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imwrite(str(output_dir / img_path.name), result_img)

    # Calculate metrics
    pcs = calculate_pcs(results['pred_scores'], results['gt_scores'])
    mase = calculate_mase(results['pred_scores'], results['gt_scores'])

    metrics = {
        'split': split,
        'num_images': len(image_files),
        'PCS': pcs,
        'MASE': mase,
        'errors': results['errors']
    }

    # Output
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({split})")
    print(f"{'='*50}")
    print(f"Images:                    {metrics['num_images']}")
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
    Prediction for a single image.

    Returns:
        Dictionary with detections and scores
    """
    model = YOLO(model_path)
    image_path = Path(image_path)

    # Prediction
    detections, img = predict_single(model, image_path, conf_threshold)

    # Keypoints and scores
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
            print(f"Score calculation failed: {e}")

    # Visualization
    score_strings = [s[0] for s, _ in score_info.get('scores', [])] or None
    result_img = draw_predictions(img, detections, scores=score_strings)

    # Score Text
    text = f"Total: {score_info['total']}"
    cv2.putText(result_img, text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Saved: {output_path}")

    # Show
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

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Evaluate Subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', '-m', required=True, help='Model path')
    eval_parser.add_argument('--data', '-d', required=True, help='Dataset path')
    eval_parser.add_argument('--split', '-s', default='test', help='Split')
    eval_parser.add_argument('--conf', type=float, default=0.25, help='Confidence')
    eval_parser.add_argument('--iou', type=float, default=0.7, help='IoU Threshold')
    eval_parser.add_argument('--output', '-o', help='Output folder')
    eval_parser.add_argument('--write', action='store_true', help='Save images')

    # Predict Subcommand
    pred_parser = subparsers.add_parser('predict', help='Single image prediction')
    pred_parser.add_argument('--model', '-m', required=True, help='Model path')
    pred_parser.add_argument('--image', '-i', required=True, help='Image path')
    pred_parser.add_argument('--conf', type=float, default=0.25, help='Confidence')
    pred_parser.add_argument('--output', '-o', help='Output path')
    pred_parser.add_argument('--show', action='store_true', help='Show image')

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
        print(f"\nResult:")
        print(f"  Darts found: {len(result['keypoints']['darts'])}")
        print(f"  Calibration points: {len(result['keypoints']['calibration'])}")
        print(f"  Scores: {result['scores']}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
