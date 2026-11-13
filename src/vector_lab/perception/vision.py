"""Deterministic computer-vision feature extraction and multi-sensor perception fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vector_lab.types import Blackboard, CameraFrame, DetectedFace, DetectedMarker, RangeScan


@dataclass(frozen=True, slots=True)
class VisionPerceptionReport:
    """Structured perceptual summary extracted from camera pixels and ToF range scans."""

    visual_edge_density: float
    fiducial_contrast_score: float
    forward_obstacle_dist_m: float
    faces: tuple[DetectedFace, ...]
    markers: tuple[DetectedMarker, ...]
    charger_visible: bool


class VisionProcessor:
    """Lightweight deterministic CV & rangefinder fusion pipeline updating the BT blackboard."""

    def __init__(self, edge_gradient_threshold: float = 22.0) -> None:
        self.edge_gradient_threshold = edge_gradient_threshold

    def compute_edge_density(self, frame: CameraFrame) -> float:
        """Compute normalized Sobel-style gradient edge density over the grayscale camera buffer."""
        img = frame.image.astype(np.float32)
        if img.shape[0] < 3 or img.shape[1] < 3:
            return 0.0
        gx = np.abs(img[:, 1:] - img[:, :-1])[:-1, :]
        gy = np.abs(img[1:, :] - img[:-1, :])[:, :-1]
        grad_mag = np.hypot(gx, gy)
        return float(np.mean(grad_mag >= self.edge_gradient_threshold))

    @staticmethod
    def compute_fiducial_contrast(frame: CameraFrame) -> float:
        """Measure local peak-to-trough contrast in detected fiducial/face ROIs or center band."""
        img = frame.image.astype(np.float32)
        if frame.markers:
            x, y, w, h = frame.markers[0].bbox_xywh
            roi = img[y : y + h, x : x + w]
            if roi.size > 0:
                return float((np.max(roi) - np.min(roi)) / 255.0)
        h_mid = img.shape[0] // 2
        band = img[max(0, h_mid - 16) : min(img.shape[0], h_mid + 16), :]
        return float((np.max(band) - np.min(band)) / 255.0) if band.size > 0 else 0.0

    def process(
        self,
        frame: CameraFrame,
        scan: RangeScan,
        blackboard: Blackboard,
    ) -> VisionPerceptionReport:
        """Extract CV & ToF range features, then write structured observations to blackboard."""
        edge_density = self.compute_edge_density(frame)
        contrast = self.compute_fiducial_contrast(frame)
        forward_dist = scan.min_forward_range_m

        blackboard.camera = frame
        blackboard.scan = scan
        blackboard.visual_edge_density = round(edge_density, 4)
        blackboard.forward_obstacle_dist_m = round(forward_dist, 4)
        blackboard.detected_faces = list(frame.faces)
        blackboard.detected_markers = list(frame.markers)

        charger_visible = any(m.kind == "charger_dock" for m in frame.markers)
        return VisionPerceptionReport(
            visual_edge_density=edge_density,
            fiducial_contrast_score=contrast,
            forward_obstacle_dist_m=forward_dist,
            faces=frame.faces,
            markers=frame.markers,
            charger_visible=charger_visible,
        )
