"""
Dart Scoring Utilities

Calculates the score based on dart positions and calibration points.
Adapted from DeepDarts, extended for 5 calibration points (incl. center).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import cv2


# Dartboard numbers clockwise, starting at 12 o'clock (20)
BOARD_NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# Angle to number mapping (every 18 degrees)
# 0° = 12 o'clock = 20, then clockwise
ANGLE_TO_NUMBER = {i: BOARD_NUMBERS[i] for i in range(20)}


class DartboardGeometry:
    """Dartboard geometry according to BDO standard."""

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
            r_double: Radius to outer edge of Double (in meters)
            r_treble: Radius to outer edge of Treble (in meters)
            r_outer_bull: Radius Outer Bull (in meters)
            r_inner_bull: Radius Inner Bull / Double Bull (in meters)
            w_ring: Width of Double/Treble rings (in meters)
        """
        self.r_double = r_double
        self.r_treble = r_treble
        self.r_outer_bull = r_outer_bull
        self.r_inner_bull = r_inner_bull
        self.w_ring = w_ring

    def get_radii_ratios(self) -> Dict[str, float]:
        """Returns radii as ratios to the Double radius."""
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
    Calculates dart scores based on keypoint positions.

    Supports two modes:
    - 4 calibration points (like DeepDarts)
    - 5 calibration points (with center)
    """

    def __init__(self, geometry: Optional[DartboardGeometry] = None):
        """
        Args:
            geometry: Dartboard geometry (default: BDO standard)
        """
        self.geometry = geometry or DartboardGeometry()
        self.ratios = self.geometry.get_radii_ratios()

    def estimate_center_and_radius(
        self,
        keypoints: np.ndarray,
        has_center: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Estimates center and radius from calibration points.

        Args:
            keypoints: Array of calibration points
                       [center, k1, k2, k3, k4] if has_center=True
                       [k1, k2, k3, k4] if has_center=False
            has_center: Whether the first point is the center

        Returns:
            (center, radius) - Center and average radius
        """
        if has_center:
            center = keypoints[0]
            outer_points = keypoints[1:5]
        else:
            # Estimate center from the 4 outer points
            center = np.mean(keypoints[:4], axis=0)
            outer_points = keypoints[:4]

        # Radius as mean distance of outer points to center
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
        Transforms points into a normalized coordinate system.

        Calibration points are used to perform a perspective 
        correction and normalize points relative to the dartboard center.

        Args:
            points: Points to transform (e.g. dart positions)
            cal_keypoints: Calibration points
            has_center: Whether a center keypoint is present

        Returns:
            (transformed_points, center, radius)
        """
        center, radius = self.estimate_center_and_radius(cal_keypoints, has_center)

        # Points relative to center
        transformed = points - center

        # Normalize to radius = 1 for Double Ring
        transformed = transformed / radius

        return transformed, center, radius

    def point_to_score(
        self,
        point: np.ndarray,
        center: np.ndarray,
        radius: float
    ) -> Tuple[str, int]:
        """
        Calculates the score for a single point.

        Args:
            point: Dart position (normalized, 0-1)
            center: Board center
            radius: Board radius (Double Ring)

        Returns:
            (score_string, score_numeric)
            e.g. ("T20", 60) or ("DB", 50)
        """
        # Relative position to center
        rel = point - center

        # Distance (normalized to Double radius)
        dist = np.linalg.norm(rel) / radius

        # Angle (0° = top, clockwise)
        angle = np.arctan2(rel[0], -rel[1])  # x, -y for "top = 0°"
        angle_deg = np.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # Determine segment (20 segments of 18° each)
        # Offset of 9° because segment boundaries are at 9°, 27°, etc.
        segment_angle = (angle_deg + 9) % 360
        segment_idx = int(segment_angle / 18) % 20
        number = BOARD_NUMBERS[segment_idx]

        # Score based on distance
        ratios = self.ratios

        if dist > ratios['double_outer']:
            # Miss - outside the board
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
        Calculates scores for all darts.

        Args:
            dart_positions: Array of dart positions [(x, y), ...]
            cal_keypoints: Array of calibration points
            has_center: Whether a center keypoint is present

        Returns:
            List of (score_string, score_numeric) tuples
        """
        if len(dart_positions) == 0:
            return []

        # Determine center and radius
        center, radius = self.estimate_center_and_radius(cal_keypoints, has_center)

        # Calculate scores
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
    Convenience function to calculate scores from predictions.

    Args:
        predictions: Dictionary with 'darts' and 'calibration' keys
                    darts: [(x, y), ...] Dart positions
                    calibration: [(x, y), ...] Calibration points
        has_center: Whether a center keypoint is present
        geometry: Dartboard geometry

    Returns:
        Dictionary with scores and total score
    """
    scorer = DartScorer(geometry)

    darts = np.array(predictions.get('darts', []))
    cal = np.array(predictions.get('calibration', []))

    if len(cal) < 4:
        return {
            'scores': [],
            'total': 0,
            'error': 'Not enough calibration points'
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
    Calculates the Percent Correct Score (PCS).

    Args:
        predictions: List of predicted total scores
        ground_truth: List of actual total scores

    Returns:
        PCS in percent (0-100)
    """
    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return (correct / len(predictions)) * 100


def calculate_mase(predictions: List[int], ground_truth: List[int]) -> float:
    """
    Calculates the Mean Absolute Score Error (MASE).

    Args:
        predictions: List of predicted total scores
        ground_truth: List of actual total scores

    Returns:
        MASE (Average Absolute Error)
    """
    if len(predictions) == 0:
        return 0.0

    errors = [abs(p - g) for p, g in zip(predictions, ground_truth)]
    return sum(errors) / len(errors)
