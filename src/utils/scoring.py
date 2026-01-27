"""
Dart Scoring Utilities

Berechnet den Score basierend auf Dart-Positionen und Kalibrationspunkten.
Adaptiert von DeepDarts, erweitert für 5 Kalibrationspunkte (inkl. Center).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import cv2


# Dartboard-Nummern im Uhrzeigersinn, startend bei 12 Uhr (20)
BOARD_NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Winkel zu Nummer Mapping (alle 18°)
# 0° = 12 Uhr = 20, dann im Uhrzeigersinn
ANGLE_TO_NUMBER = {i: BOARD_NUMBERS[i] for i in range(20)}


class DartboardGeometry:
    """Dartboard-Geometrie nach BDO Standard."""

    def __init__(
        self,
        r_double: float = 0.170,
        r_treble: float = 0.1074,
        r_outer_bull: float = 0.0159,
        r_inner_bull: float = 0.00635,
        w_ring: float = 0.01
    ):
        """
        Args:
            r_double: Radius bis Außenkante Double (in Metern)
            r_treble: Radius bis Außenkante Treble (in Metern)
            r_outer_bull: Radius Outer Bull (in Metern)
            r_inner_bull: Radius Inner Bull / Double Bull (in Metern)
            w_ring: Breite der Double/Treble Ringe (in Metern)
        """
        self.r_double = r_double
        self.r_treble = r_treble
        self.r_outer_bull = r_outer_bull
        self.r_inner_bull = r_inner_bull
        self.w_ring = w_ring

    def get_radii_ratios(self) -> Dict[str, float]:
        """Gibt die Radien als Verhältnisse zum Double-Radius zurück."""
        return {
            'double_outer': 1.0,
            'double_inner': 1.0 - (self.w_ring / self.r_double),
            'treble_outer': self.r_treble / self.r_double,
            'treble_inner': (self.r_treble - self.w_ring) / self.r_double,
            'outer_bull': self.r_outer_bull / self.r_double,
            'inner_bull': self.r_inner_bull / self.r_double,
        }


class DartScorer:
    """
    Berechnet Dart-Scores basierend auf Keypoint-Positionen.

    Unterstützt zwei Modi:
    - 4 Kalibrationspunkte (wie DeepDarts)
    - 5 Kalibrationspunkte (mit Center)
    """

    def __init__(self, geometry: Optional[DartboardGeometry] = None):
        """
        Args:
            geometry: Dartboard-Geometrie (default: BDO Standard)
        """
        self.geometry = geometry or DartboardGeometry()
        self.ratios = self.geometry.get_radii_ratios()

    def estimate_center_and_radius(
        self,
        keypoints: np.ndarray,
        has_center: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Schätzt Zentrum und Radius aus den Kalibrationspunkten.

        Args:
            keypoints: Array der Kalibrationspunkte
                       [center, k1, k2, k3, k4] wenn has_center=True
                       [k1, k2, k3, k4] wenn has_center=False
            has_center: Ob der erste Punkt das Zentrum ist

        Returns:
            (center, radius) - Zentrum und mittlerer Radius
        """
        if has_center:
            center = keypoints[0]
            outer_points = keypoints[1:5]
        else:
            # Zentrum aus den 4 äußeren Punkten schätzen
            center = np.mean(keypoints[:4], axis=0)
            outer_points = keypoints[:4]

        # Radius als mittlere Distanz der äußeren Punkte zum Zentrum
        distances = np.linalg.norm(outer_points - center, axis=1)
        radius = np.mean(distances)

        return center, radius

    def transform_to_normalized(
        self,
        points: np.ndarray,
        cal_keypoints: np.ndarray,
        has_center: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Transformiert Punkte in ein normalisiertes Koordinatensystem.

        Die Kalibrationspunkte werden verwendet, um eine perspektivische
        Korrektur durchzuführen und die Punkte relativ zum Dartboard-Zentrum
        zu normalisieren.

        Args:
            points: Zu transformierende Punkte (z.B. Dart-Positionen)
            cal_keypoints: Kalibrationspunkte
            has_center: Ob ein Center-Keypoint vorhanden ist

        Returns:
            (transformed_points, center, radius)
        """
        center, radius = self.estimate_center_and_radius(cal_keypoints, has_center)

        # Punkte relativ zum Zentrum
        transformed = points - center

        # Normalisieren auf Radius = 1 für Double-Ring
        transformed = transformed / radius

        return transformed, center, radius

    def point_to_score(
        self,
        point: np.ndarray,
        center: np.ndarray,
        radius: float
    ) -> Tuple[str, int]:
        """
        Berechnet den Score für einen einzelnen Punkt.

        Args:
            point: Dart-Position (normalisiert, 0-1)
            center: Board-Zentrum
            radius: Board-Radius (Double-Ring)

        Returns:
            (score_string, score_numeric)
            z.B. ("T20", 60) oder ("DB", 50)
        """
        # Relative Position zum Zentrum
        rel = point - center

        # Distanz (normalisiert auf Double-Radius)
        dist = np.linalg.norm(rel) / radius

        # Winkel (0° = oben, im Uhrzeigersinn)
        angle = np.arctan2(rel[0], -rel[1])  # x, -y für "oben = 0°"
        angle_deg = np.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # Segment bestimmen (20 Segmente à 18°)
        # Offset von 9° weil die Segmentgrenzen bei 9°, 27°, etc. liegen
        segment_angle = (angle_deg + 9) % 360
        segment_idx = int(segment_angle / 18) % 20
        number = BOARD_NUMBERS[segment_idx]

        # Score basierend auf Distanz
        ratios = self.ratios

        if dist > ratios['double_outer']:
            # Miss - außerhalb des Boards
            return "0", 0

        elif dist <= ratios['inner_bull']:
            # Double Bull (Bullseye)
            return "DB", 50

        elif dist <= ratios['outer_bull']:
            # Single Bull
            return "B", 25

        elif dist <= ratios['double_outer'] and dist > ratios['double_inner']:
            # Double Ring
            return f"D{number}", number * 2

        elif dist <= ratios['treble_outer'] and dist > ratios['treble_inner']:
            # Treble Ring
            return f"T{number}", number * 3

        else:
            # Single
            return str(number), number

    def calculate_scores(
        self,
        dart_positions: np.ndarray,
        cal_keypoints: np.ndarray,
        has_center: bool = True
    ) -> List[Tuple[str, int]]:
        """
        Berechnet Scores für alle Darts.

        Args:
            dart_positions: Array mit Dart-Positionen [(x, y), ...]
            cal_keypoints: Array mit Kalibrationspunkten
            has_center: Ob ein Center-Keypoint vorhanden ist

        Returns:
            Liste von (score_string, score_numeric) Tupeln
        """
        if len(dart_positions) == 0:
            return []

        # Zentrum und Radius bestimmen
        center, radius = self.estimate_center_and_radius(cal_keypoints, has_center)

        # Scores berechnen
        scores = []
        for dart in dart_positions:
            score_str, score_num = self.point_to_score(dart, center, radius)
            scores.append((score_str, score_num))

        return scores


def get_dart_scores(
    predictions: Dict,
    has_center: bool = True,
    geometry: Optional[DartboardGeometry] = None
) -> Dict:
    """
    Convenience-Funktion zur Score-Berechnung aus Predictions.

    Args:
        predictions: Dictionary mit 'darts' und 'calibration' Keys
                    darts: [(x, y), ...] Dart-Positionen
                    calibration: [(x, y), ...] Kalibrationspunkte
        has_center: Ob ein Center-Keypoint vorhanden ist
        geometry: Dartboard-Geometrie

    Returns:
        Dictionary mit Scores und Gesamtpunktzahl
    """
    scorer = DartScorer(geometry)

    darts = np.array(predictions.get('darts', []))
    cal = np.array(predictions.get('calibration', []))

    if len(cal) < 4:
        return {
            'scores': [],
            'total': 0,
            'error': 'Nicht genug Kalibrationspunkte'
        }

    scores = scorer.calculate_scores(darts, cal, has_center)

    return {
        'scores': scores,
        'score_strings': [s[0] for s in scores],
        'score_values': [s[1] for s in scores],
        'total': sum(s[1] for s in scores)
    }


def calculate_pcs(predictions: List[int], ground_truth: List[int]) -> float:
    """
    Berechnet den Percent Correct Score (PCS).

    Args:
        predictions: Liste der vorhergesagten Gesamtscores
        ground_truth: Liste der tatsächlichen Gesamtscores

    Returns:
        PCS in Prozent (0-100)
    """
    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return (correct / len(predictions)) * 100


def calculate_mase(predictions: List[int], ground_truth: List[int]) -> float:
    """
    Berechnet den Mean Absolute Score Error (MASE).

    Args:
        predictions: Liste der vorhergesagten Gesamtscores
        ground_truth: Liste der tatsächlichen Gesamtscores

    Returns:
        MASE (durchschnittlicher absoluter Fehler)
    """
    if len(predictions) == 0:
        return 0.0

    errors = [abs(p - g) for p, g in zip(predictions, ground_truth)]
    return sum(errors) / len(errors)
