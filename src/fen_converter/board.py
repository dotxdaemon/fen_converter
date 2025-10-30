from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

FILES = "abcdefgh"
RANKS = "87654321"


class BoardExtractionError(RuntimeError):
    """Raised when the chessboard cannot be detected inside the source image."""


@dataclasses.dataclass(frozen=True)
class SquareImage:
    square: str
    image: np.ndarray

    def as_ascii(self, width: int = 16, height: int = 16) -> str:
        """Return a lightweight ASCII representation used for manual labeling."""

        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, (width, height), interpolation=cv2.INTER_AREA)
        gradient_x = np.abs(np.diff(resized.astype("float32"), axis=1))
        gradient_y = np.abs(np.diff(resized.astype("float32"), axis=0))
        energy = np.pad(gradient_x, ((0, 0), (1, 0)), mode="constant")
        energy += np.pad(gradient_y, ((1, 0), (0, 0)), mode="constant")
        lut = np.asarray(list(" .:-=+*#%@"))
        indices = np.clip((energy / (energy.max() + 1e-6) * (len(lut) - 1)).round().astype(int), 0, len(lut) - 1)
        lines = ["".join(lut[row]) for row in indices]
        return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class _DetectedBoard:
    image: np.ndarray
    transform: np.ndarray
    corners: np.ndarray


def detect_board(path: Path, output_size: int = 800) -> _DetectedBoard:
    """Detect the chessboard in *path* and return a rectified image."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise BoardExtractionError("No contours detected in the source image.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_quad: List[np.ndarray] = []
    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            board_quad = approx.reshape(4, 2)
            break

    if len(board_quad) != 4:
        raise BoardExtractionError("Unable to locate a four-point contour for the chessboard.")

    def _sort_clockwise(points: np.ndarray) -> np.ndarray:
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        top_left = points[np.argmin(s)]
        bottom_right = points[np.argmax(s)]
        top_right = points[np.argmin(diff)]
        bottom_left = points[np.argmax(diff)]
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    ordered = _sort_clockwise(board_quad.astype("float32"))
    dst = np.array(
        [[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, transform, (output_size, output_size))

    return _DetectedBoard(image=warped, transform=transform, corners=ordered)


def extract_square_images(board: np.ndarray) -> Dict[str, SquareImage]:
    """Return square images from a rectified chessboard image."""

    size = board.shape[0]
    step = size // 8
    squares: Dict[str, SquareImage] = {}
    for rank_index, rank in enumerate(RANKS):
        for file_index, file in enumerate(FILES):
            y0 = rank_index * step
            y1 = (rank_index + 1) * step
            x0 = file_index * step
            x1 = (file_index + 1) * step
            crop = board[y0:y1, x0:x1].copy()
            name = f"{file}{rank}"
            squares[name] = SquareImage(square=name, image=crop)
    return squares
