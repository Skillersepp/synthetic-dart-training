"""
Dart Scoring Utilities

Calculates the score based on dart positions and calibration points.
Uses Homography transformation based on board physical layout.
"""

import math
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union

class DartboardLayout:
    """
    Represents the physical layout of a standard WDF dartboard.
    Provides methods for coordinate validation and field identification.
    """

    # Segment Scores (starting from 6 and going counter-clockwise)
    SEGMENTS = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
    
    SEGMENT_WIDTH_RAD = math.radians(18)  # Each segment is 18 degrees wide = 2π/20 radians

    # Dimensions in mm (WDF Standards)
    R_INNER_BULL = 6.35
    R_OUTER_BULL = 15.9
    R_INNER_TREBLE = 97.4
    R_OUTER_TREBLE = 107.4
    R_INNER_DOUBLE = 160.0
    R_OUTER_DOUBLE = 170.0

    # Wire diameters in mm
    D_INNER_BULL_WIRE = 1.2
    D_OUTER_BULL_WIRE = 1.2
    D_TREBLE_WIRE = 1.0
    D_DOUBLE_WIRE = 1.0
    
    # Default dart tip radius in mm (used for collision checks)
    DEFAULT_R_TIP = 1.1

    def __init__(self, r_tip: float = DEFAULT_R_TIP):
        self.r_tip = r_tip
        self.invalid_intervals = self._calculate_invalid_intervals()

    def _calculate_invalid_intervals(self):
        """
        Pre-calculates the invalid radius intervals where a dart would hit a wire.
        Returns a list of (start, end) tuples in mm.
        """
        # Short aliases for readability
        rib = self.R_INNER_BULL
        rob = self.R_OUTER_BULL
        rit = self.R_INNER_TREBLE
        rot = self.R_OUTER_TREBLE
        rid = self.R_INNER_DOUBLE
        rod = self.R_OUTER_DOUBLE

        dib = self.D_INNER_BULL_WIRE
        dob = self.D_OUTER_BULL_WIRE
        dt = self.D_TREBLE_WIRE
        dd = self.D_DOUBLE_WIRE
        
        rt = self.r_tip

        # Define invalid intervals (open intervals)
        # Each tuple is (start, end)
        return [
            (rib - rt, rib + dib + rt),
            (rob - rt, rob + dob + rt),
            (rit - dt - rt, rit + rt),
            (rot - dt - rt, rot + rt),
            (rid - dd - rt, rid + rt),
            (rod - dd - rt, rod + rt)
        ]

    def validate_radius(self, radius_m: float) -> float:
        """
        Validate and adjust the radius to ensure the dart doesn't hit the wire.
        
        Args:
            radius_m: Radius in meters.
            
        Returns:
            Adjusted radius in meters.
        """
        r_mm = radius_m * 1000.0
        
        for start, end in self.invalid_intervals:
            if start < r_mm < end:
                # Radius is in invalid interval
                dist_to_start = abs(r_mm - start)
                dist_to_end = abs(r_mm - end)
                
                if dist_to_start < dist_to_end:
                    r_mm = start
                else:
                    r_mm = end
                
                # Since intervals are open, setting to boundary is fine
                break 
        
        return r_mm / 1000.0

    def validate_angle(self, radius_m: float, angle_rad: float) -> float:
        """
        Validate and adjust the angle to ensure the dart doesn't hit the radial wires.
        
        Args:
            radius_m: Radius in meters.
            angle_rad: Angle in radians.
            
        Returns:
            Adjusted angle in radians.
        """
        r_mm = radius_m * 1000.0
        
        # Only check if radius is within the specified interval [15.8, 180.0]
        if not (15.8 <= r_mm <= 180.0):
            return angle_rad
            
        # Calculate required angular margin
        # margin = 0.6mm (half wire) + r_tip
        margin_mm = 0.6 + self.r_tip
        
        # Calculate angular half-width of the exclusion zone
        # sin(dtheta) = margin / r
        # For small angles, sin(x) approx x, but we use asin for correctness
        if margin_mm >= r_mm:
            # Should not happen in valid range, but safety check
            return angle_rad
            
        dtheta = math.asin(margin_mm / r_mm)
        
        # Dartboard geometry:
        # 20 segments, each 18 degrees (pi/10 radians)
        # 0 degrees is at the center of the "6" segment (Right)
        # Wires are at +/- 9 degrees (pi/20) from the center of each segment
        
        segment_angle = 2 * math.pi / 20 # 18 degrees
        half_segment = segment_angle / 2 # 9 degrees
        
        # Shift angle so that wires are at 0, segment_angle, 2*segment_angle...
        # Original wires: +/- 9 deg, +/- 27 deg...
        # Add 9 deg (half_segment) -> wires at 0, 18, 36...
        angle_shifted = angle_rad + half_segment
        
        # Modulo segment angle to find position within the "wire-to-wire" interval
        angle_mod = angle_shifted % segment_angle
        
        # Distance to the nearest wire (which is at 0 or segment_angle in this shifted space)
        dist_to_wire = min(angle_mod, segment_angle - angle_mod)
        
        if dist_to_wire < dtheta:
            # Collision with wire
            correction = dtheta - dist_to_wire
            # Determine direction to push
            if angle_mod < half_segment:
                # Closer to the "left" wire (0 in shifted space)
                # Push away (increase angle)
                angle_rad += correction
            else:
                # Closer to the "right" wire (segment_angle in shifted space)
                # Push away (decrease angle)
                angle_rad -= correction
                
        return angle_rad

    def get_score_from_polar(self, radius_m: float, angle_rad: float):
        """
        Determines the dart score from polar coordinates.

        Args:
            radius_m: Radius in meters.
            angle_rad: Angle in radians.
            
        Returns:
            Dart score.
        """

        r_mm = radius_m * 1000.0

        # ---------- special cases ----------
        if r_mm > self.R_OUTER_DOUBLE:
            return 0   # Miss

        if r_mm <= self.R_INNER_BULL:
            return 50  # Bullseye

        if r_mm <= self.R_OUTER_BULL:
            return 25  # Single Bull


        # ---------- Determine segment ----------
        # Normalize angle to [0, 2π)
        theta = angle_rad % (2 * math.pi)

        # Offset by half segment to align with segment
        theta_shifted = theta + self.SEGMENT_WIDTH_RAD / 2

        index = int(theta_shifted / self.SEGMENT_WIDTH_RAD) % 20

        base_value = self.SEGMENTS[index]

        # ---------- Treble / Double ----------
        if self.R_INNER_TREBLE <= r_mm <= self.R_OUTER_TREBLE:
            return 3 * base_value

        if self.R_INNER_DOUBLE <= r_mm <= self.R_OUTER_DOUBLE:
            return 2 * base_value

        # ---------- Single ----------
        return base_value

    @classmethod
    def get_keypoint_positions(cls) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Returns the standard physical positions of keypoints.
        Returns a dict: {name: (x_mm, y_mm, angle_rad, radius_mm)}
        
        Keypoints:
          k1: Outer Double Ring, between 5 and 20 (99°)
          k2: Outer Double Ring, between 17 and 3 (279°)
          k3: Outer Double Ring, between 8 and 11 (189°)
          k4: Outer Double Ring, between 13 and 6 (9°)
          center: (0,0)
          
          Optional Triple Ring Keypoints (KT1-KT4) at same angles but Triple radius.
        """
        positions = {}
        
        # Angles in degrees and radians
        # K4 (D13/D6): 9 deg
        # K1 (D5/D20): 99 deg
        # K3 (D8/D11): 189 deg
        # K2 (D17/D3): 279 deg
        
        angles_deg = {
            'cal_k4': 9,
            'cal_k1': 99,
            'cal_k3': 189,
            'cal_k2': 279
        }
        
        r_double = cls.R_OUTER_DOUBLE
        r_triple = cls.R_OUTER_TREBLE
        
        # Center
        positions['cal_center'] = (0.0, 0.0, 0.0, 0.0)
        
        for name, deg in angles_deg.items():
            rad = math.radians(deg)
            
            # Double Ring Keypoints
            x = r_double * math.cos(rad)
            y = r_double * math.sin(rad)
            positions[name] = (x, y, rad, r_double)
            
            # Triple Ring Keypoints (KT1-KT4)
            # Assuming naming convention cal_kt1, etc. if they exist
            kt_name = name.replace('k', 'kt')
            x_t = r_triple * math.cos(rad)
            y_t = r_triple * math.sin(rad)
            positions[kt_name] = (x_t, y_t, rad, r_triple)
            
        return positions


class DartScorer:
    """
    Calculates dart scores based on keypoint positions using Homography.
    """

    def __init__(self, geometry: Optional[Any] = None):
        """
        Args:
            geometry: Legacy argument, ignored.
        """
        self.layout = DartboardLayout()

    def compute_homography(self, detected_keypoints: Dict[str, Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Computes Homography matrix from detected keypoints to physical board coordinates (mm).
        
        Args:
            detected_keypoints: Dict of {name: (x, y)} in image coordinates (pixels or normalized).
                                Names should match 'cal_center', 'cal_k1', etc.
        
        Returns:
            3x3 Homography matrix or None if insufficient points.
        """
        physical_kps = self.layout.get_keypoint_positions()
        
        src_points = [] # Image points
        dst_points = [] # Physical points (mm)
        
        # Mapping from possible keys to standard keys
        # The user might pass 'center', 'k1' etc. or 'cal_center', 'cal_k1'.
        # We normalize to 'cal_...'
        
        for name, (x, y) in detected_keypoints.items():
            if x is None or y is None:
                continue

            norm_name = name if name.startswith('cal_') else f'cal_{name}'
            
            if norm_name in physical_kps:
                # Get physical coord (x_mm, y_mm)
                phys_x, phys_y, _, _ = physical_kps[norm_name]
                
                src_points.append([x, y])
                dst_points.append([phys_x, phys_y])
                
        if len(src_points) < 4:
            # Not enough points for Homography (need 4)
            return None
            
        src_pts = np.array(src_points, dtype=np.float32)
        dst_pts = np.array(dst_points, dtype=np.float32)
        
        # Compute Homography
        # transforming src (image) -> dst (physical mm)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        return H

    def get_score(
        self,
        point: Tuple[float, float],
        H: np.ndarray
    ) -> Tuple[int, float, float]:
        """
        Calculates score for a point using the homography.
        
        Args:
            point: (x, y) in image coordinates.
            H: Homography matrix.
            
        Returns:
            (score, radius_m, angle_rad)
        """
        if H is None:
            return 0, 0.0, 0.0
            
        # Perspective transform of the point
        pt_np = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_np, H)
        
        x_mm = transformed[0][0][0]
        y_mm = transformed[0][0][1]
        
        # Convert to polar
        radius_mm = math.sqrt(x_mm**2 + y_mm**2)
        angle_rad = math.atan2(y_mm, x_mm)
        
        # Ensure positive angle [0, 2pi]
        if angle_rad < 0:
            angle_rad += 2 * math.pi
            
        radius_m = radius_mm / 1000.0
        
        score = self.layout.get_score_from_polar(radius_m, angle_rad)
        
        return score, radius_m, angle_rad
        
    def calculate_scores(
        self,
        dart_positions: List[Tuple[float, float]],
        cal_keypoints: Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]]],
        has_center: bool = True # Legacy arg
    ) -> List[Tuple[str, int]]:
        """
        Calculates scores for all darts.

        Args:
            dart_positions: List of (x, y)
            cal_keypoints: Dict of {name: (x, y)} OR List of points [center, k1, k2, k3, k4]
            has_center: Legacy arg, ignored.
                       
        Returns:
            List of (score_string, score_value)
        """
        # Convert list to dict if necessary (legacy support)
        kp_dict = {}
        if isinstance(cal_keypoints, (list, np.ndarray)):
            # Assumes order: center, k1, k2, k3, k4
            names = ['cal_center', 'cal_k1', 'cal_k2', 'cal_k3', 'cal_k4']
            for i, pt in enumerate(cal_keypoints):
                if i >= len(names): break
                
                valid = False
                if pt is not None:
                    if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2:
                        valid = True
                
                if valid:
                    kp_dict[names[i]] = (float(pt[0]), float(pt[1]))
        elif isinstance(cal_keypoints, dict):
            kp_dict = cal_keypoints

        H = self.compute_homography(kp_dict)
        
        scores = []
        for pt in dart_positions:
            score_val, r, theta = self.get_score(pt, H)
            score_str = self._format_score(score_val, r * 1000.0) # r in meters -> mm
            scores.append((score_str, score_val))
            
        return scores

    def _format_score(self, score: int, r_mm: float) -> str:
        layout = self.layout
        if score == 0:
            return "0"
        if score == 50:
            return "DB"
            
        # Check regions
        if r_mm <= layout.R_OUTER_BULL:
            return "B" if score == 25 else "DB"
            
        # Determine multiplier
        multiplier = 1
        if layout.R_INNER_DOUBLE <= r_mm <= layout.R_OUTER_DOUBLE:
            multiplier = 2
        elif layout.R_INNER_TREBLE <= r_mm <= layout.R_OUTER_TREBLE:
            multiplier = 3
            
        base = score // multiplier
        prefix = {1: "", 2: "D", 3: "T"}[multiplier]
        return f"{prefix}{base}"


def get_dart_scores(
    predictions: Dict,
    has_center: bool = True,
    geometry: Optional[Any] = None
) -> Dict:
    """
    Convenience function to calculate scores from predictions.
    """
    scorer = DartScorer()
    
    darts = predictions.get('darts', [])
    cal = predictions.get('calibration', {})
    
    scores = scorer.calculate_scores(darts, cal)
    
    return {
        'scores': scores,
        'score_strings': [s[0] for s in scores],
        'score_values': [s[1] for s in scores],
        'total': sum(s[1] for s in scores)
    }

def calculate_pcs(predictions: List[int], ground_truth: List[int]) -> float:
    """Calculates the Percent Correct Score (PCS)."""
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return (correct / len(predictions)) * 100

def calculate_mase(predictions: List[int], ground_truth: List[int]) -> float:
    """Calculates the Mean Absolute Score Error (MASE)."""
    if len(predictions) == 0:
        return 0.0
    errors = [abs(p - g) for p, g in zip(predictions, ground_truth)]
    return sum(errors) / len(errors)
