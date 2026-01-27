"""
Visualization Utilities

Funktionen zum Zeichnen von Predictions und Dartboard-Overlays.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# Farben (BGR Format für OpenCV)
COLORS = {
    'dart': (0, 255, 255),       # Gelb/Cyan
    'cal_center': (0, 255, 0),   # Grün
    'cal_k1': (255, 0, 0),       # Blau
    'cal_k2': (255, 0, 0),       # Blau
    'cal_k3': (255, 0, 0),       # Blau
    'cal_k4': (255, 0, 0),       # Blau
    'bbox': (0, 165, 255),       # Orange
    'text': (255, 255, 255),     # Weiß
    'board_rings': (128, 128, 128),  # Grau
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
    Zeichnet einen Keypoint auf das Bild.

    Args:
        img: Bild (wird modifiziert)
        x, y: Normalisierte Koordinaten (0-1)
        color: BGR Farbe
        radius: Punkt-Radius
        thickness: -1 für gefüllt
        label: Optional Label-Text

    Returns:
        Modifiziertes Bild
    """
    h, w = img.shape[:2]
    px, py = int(x * w), int(y * h)

    # Punkt zeichnen
    cv2.circle(img, (px, py), radius, color, thickness)

    # Label zeichnen
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
    Zeichnet eine Bounding Box auf das Bild.

    Args:
        img: Bild (wird modifiziert)
        x_center, y_center: Normalisierte Zentrum-Koordinaten
        width, height: Normalisierte Größe
        color: BGR Farbe
        thickness: Liniendicke
        label: Optional Label-Text

    Returns:
        Modifiziertes Bild
    """
    h, w = img.shape[:2]

    # Konvertiere zu Pixel-Koordinaten
    x1 = int((x_center - width / 2) * w)
    y1 = int((y_center - height / 2) * h)
    x2 = int((x_center + width / 2) * w)
    y2 = int((y_center + height / 2) * h)

    # Box zeichnen
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Label zeichnen
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
    Zeichnet YOLO-Detections auf ein Bild.

    Args:
        img: Eingabebild
        detections: Liste von Detections, jede mit:
                   - class_id: int
                   - x_center, y_center, width, height: float (normalisiert)
                   - confidence: float
        draw_boxes: Bounding Boxes zeichnen
        draw_points: Keypoints zeichnen
        draw_labels: Labels zeichnen
        scores: Optional Liste von Score-Strings für Darts

    Returns:
        Bild mit eingezeichneten Detections
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

        # Keypoint (Zentrum der Box)
        if draw_points:
            # Score-Label für Darts
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
    Zeichnet ein Dartboard-Overlay (Ringe und Segmentlinien).

    Args:
        img: Eingabebild
        center: Normalisiertes Zentrum (x, y)
        radius: Normalisierter Radius (Double-Ring)
        color: BGR Farbe
        thickness: Liniendicke

    Returns:
        Bild mit Dartboard-Overlay
    """
    img = img.copy()
    h, w = img.shape[:2]

    # Pixel-Koordinaten
    cx, cy = int(center[0] * w), int(center[1] * h)
    r_px = int(radius * min(w, h))

    # Radien-Verhältnisse (BDO Standard)
    ratios = {
        'double_outer': 1.0,
        'double_inner': 0.941,
        'treble_outer': 0.632,
        'treble_inner': 0.573,
        'outer_bull': 0.094,
        'inner_bull': 0.037,
    }

    # Ringe zeichnen
    for name, ratio in ratios.items():
        r = int(r_px * ratio)
        cv2.circle(img, (cx, cy), r, color, thickness)

    # Segment-Linien (alle 18°)
    for i in range(20):
        angle = np.radians(i * 18 - 9)  # -9° Offset für Segmentgrenzen
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
    Zeichnet Predictions und Ground Truth zum Vergleich.

    Args:
        img: Eingabebild
        predictions: Liste von Predictions
        ground_truth: Liste von Ground Truth Detections
        pred_color: Farbe für Predictions (default: Grün)
        gt_color: Farbe für Ground Truth (default: Rot)

    Returns:
        Vergleichsbild
    """
    img = img.copy()

    # Ground Truth (als X)
    for gt in ground_truth:
        x, y = gt['x_center'], gt['y_center']
        h, w = img.shape[:2]
        px, py = int(x * w), int(y * h)
        size = 8
        cv2.line(img, (px - size, py - size), (px + size, py + size), gt_color, 2)
        cv2.line(img, (px - size, py + size), (px + size, py - size), gt_color, 2)

    # Predictions (als Kreis)
    for pred in predictions:
        x, y = pred['x_center'], pred['y_center']
        h, w = img.shape[:2]
        px, py = int(x * w), int(y * h)
        cv2.circle(img, (px, py), 6, pred_color, 2)

    # Legende
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
    Speichert ein Bild mit eingezeichneten Predictions.

    Args:
        img: Eingabebild
        output_path: Ausgabepfad
        detections: Liste von Detections
        scores: Optional Liste von Score-Strings
    """
    result = draw_predictions(img, detections, scores=scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
