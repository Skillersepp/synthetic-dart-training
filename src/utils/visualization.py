"""
Visualization Utilities

Functions for drawing predictions and dartboard overlays.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# Colors (BGR format for OpenCV)
COLORS = {
    'dart': (0, 255, 255),       # Yellow/Cyan
    'cal_center': (0, 255, 0),   # Green
    'cal_k1': (255, 0, 0),       # Blue
    'cal_k2': (255, 0, 0),       # Blue
    'cal_k3': (255, 0, 0),       # Blue
    'cal_k4': (255, 0, 0),       # Blue
    'bbox': (0, 165, 255),       # Orange
    'text': (255, 255, 255),     # White
    'board_rings': (128, 128, 128),  # Gray
}

CLASS_NAMES = {
    0: 'dart',
    1: 'cal_center',
    2: 'cal_k1',
    3: 'cal_k2',
    4: 'cal_k3',
    5: 'cal_k4',
}


def draw_keypoint(
    img: np.ndarray,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    radius: int = 5,
    thickness: int = -1,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draws a keypoint on the image.

    Args:
        img: Image (will be modified)
        x, y: Normalized coordinates (0-1)
        color: BGR color
        radius: Point radius
        thickness: -1 for filled
        label: Optional label text

    Returns:
        Modified image
    """
    h, w = img.shape[:2]
    px, py = int(x * w), int(y * h)

    # Draw point
    cv2.circle(img, (px, py), radius, color, thickness)

    # Draw label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        cv2.putText(img, label, (px + 8, py - 5), font, font_scale, color, 1)

    return img


def draw_bbox(
    img: np.ndarray,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Draws a bounding box on the image.

    Args:
        img: Image (will be modified)
        x_center, y_center: Normalized center coordinates
        width, height: Normalized dimensions
        color: BGR color
        thickness: Line thickness
        label: Optional label text

    Returns:
        Modified image
    """
    h, w = img.shape[:2]

    # Convert to pixel coordinates
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Draw label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, font_scale, (0, 0, 0), 1)

    return img


def draw_predictions(
    img: np.ndarray,
    detections: List[Dict],
    draw_boxes: bool = True,
    draw_points: bool = True,
    draw_labels: bool = True,
    scores: Optional[List[str]] = None
) -> np.ndarray:
    """
    Draws YOLO detections on an image.

    Args:
        img: Input image
        detections: List of detections, each with:
                   - class_id: int
                   - x_center, y_center, width, height: float (normalized)
                   - confidence: float
        draw_boxes: Draw bounding boxes
        draw_points: Draw keypoints
        draw_labels: Draw labels
        scores: Optional list of score strings for darts

    Returns:
        Image with drawn detections
    """
    img = img.copy()
    dart_idx = 0

    for det in detections:
        class_id = det['class_id']
        x, y = det['x_center'], det['y_center']
        w, h = det['width'], det['height']
        conf = det.get('confidence', 1.0)

        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        color = COLORS.get(class_name, (255, 255, 255))

        # Bounding Box
        if draw_boxes:
            label = f"{class_name} {conf:.2f}" if draw_labels else None
            draw_bbox(img, x, y, w, h, color, thickness=1, label=label)

        # Keypoint (Center of the box)
        if draw_points:
            # Score label for darts
            if class_id == 0 and scores and dart_idx < len(scores):
                point_label = scores[dart_idx]
                dart_idx += 1
            elif draw_labels:
                point_label = class_name
            else:
                point_label = None

            draw_keypoint(img, x, y, color, radius=4, label=point_label)

    return img


def draw_dartboard_overlay(
    img: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    color: Tuple[int, int, int] = (128, 128, 128),
    thickness: int = 1
) -> np.ndarray:
    """
    Draws a dartboard overlay (rings and segment lines).

    Args:
        img: Input image
        center: Normalized center (x, y)
        radius: Normalized radius (Double Ring)
        color: BGR color
        thickness: Line thickness

    Returns:
        Image with dartboard overlay
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Pixel coordinates
    cx, cy = int(center[0] * w), int(center[1] * h)
    r_px = int(radius * min(w, h))

    # Radii ratios (BDO Standard)
    ratios = {
        'double_outer': 1.0,
        'double_inner': 0.941,
        'treble_outer': 0.632,
        'treble_inner': 0.573,
        'outer_bull': 0.094,
        'inner_bull': 0.037,
    }

    # Draw rings
    for name, ratio in ratios.items():
        r = int(r_px * ratio)
        cv2.circle(img, (cx, cy), r, color, thickness)

    # Segment lines (every 18 degrees)
    for i in range(20):
        angle = np.radians(i * 18 - 9)  # -9° offset for segment boundaries
        x_inner = int(cx + r_px * ratios['outer_bull'] * np.sin(angle))
        y_inner = int(cy - r_px * ratios['outer_bull'] * np.cos(angle))
        x_outer = int(cx + r_px * np.sin(angle))
        y_outer = int(cy - r_px * np.cos(angle))
        cv2.line(img, (x_inner, y_inner), (x_outer, y_outer), color, thickness)

    return img


def draw_ground_truth_comparison(
    img: np.ndarray,
    predictions: List[Dict],
    ground_truth: List[Dict],
    pred_color: Tuple[int, int, int] = (0, 255, 0),
    gt_color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Draws predictions and ground truth for comparison.

    Args:
        img: Input image
        predictions: List of predictions
        ground_truth: List of ground truth detections
        pred_color: Color for predictions (default: Green)
        gt_color: Color for ground truth (default: Red)

    Returns:
        Comparison image
    """
    img = img.copy()

    # Ground Truth (as X)
    for gt in ground_truth:
        x, y = gt['x_center'], gt['y_center']
        h, w = img.shape[:2]
        px, py = int(x * w), int(y * h)
        size = 8
        cv2.line(img, (px - size, py - size), (px + size, py + size), gt_color, 2)
        cv2.line(img, (px - size, py + size), (px + size, py - size), gt_color, 2)

    # Predictions (as Circle)
    for pred in predictions:
        x, y = pred['x_center'], pred['y_center']
        h, w = img.shape[:2]
        px, py = int(x * w), int(y * h)
        cv2.circle(img, (px, py), 6, pred_color, 2)

    # Legend
    cv2.putText(img, "Pred (O)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
    cv2.putText(img, "GT (X)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gt_color, 2)

    return img


def save_prediction_image(
    img: np.ndarray,
    output_path: Path,
    detections: List[Dict],
    scores: Optional[List[str]] = None
) -> None:
    """
    Saves an image with drawn predictions.

    Args:
        img: Input image
        output_path: Output path
        detections: List of detections
        scores: Optional list of score strings
    """
    result = draw_predictions(img, detections, scores=scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
