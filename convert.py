"""Convert a chessboard screenshot to FEN notation."""

import argparse
import pickle
from typing import Dict, TYPE_CHECKING, Tuple

import chess

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    import numpy as np

CONTOUR_FILE = "contours.dat"
REQUIRED_CONTOURS = ["P", "N", "B", "R", "Q", "K"]
TEMPLATE_FILE = "piece_templates.npz"
CLASSIFIER_IMAGE_SIZE = 64

_TEMPLATES: Tuple["np.ndarray", "np.ndarray"] | None = None


def _normalize_rows(arr: "np.ndarray") -> "np.ndarray":
    """Normalize each row vector to have zero mean and unit variance."""

    import numpy as np

    arr = arr.astype(np.float32, copy=False)
    rows = np.atleast_2d(arr)
    means = rows.mean(axis=1, keepdims=True)
    stds = rows.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-6, 1.0, stds)
    normalized = (rows - means) / stds
    if arr.ndim == 1:
        return normalized[0]
    return normalized


def load_contours() -> Dict[str, "np.ndarray"]:
    """Load the pre-generated piece contours from a file.

    Provides clearer error messages when the file is missing or corrupted and
    validates that the expected contour symbols are present.
    """
    import numpy as np  # Imported lazily to avoid hard dependency

    try:
        with open(CONTOUR_FILE, "rb") as f:
            contours = pickle.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Contour file '{CONTOUR_FILE}' not found. "
            "Please run generate_contours.py first to create it."
        ) from exc
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            f"Contour file '{CONTOUR_FILE}' is corrupted. "
            "Please regenerate it by running generate_contours.py."
        ) from exc

    if not isinstance(contours, dict) or not all(symbol in contours for symbol in REQUIRED_CONTOURS):
        raise RuntimeError(
            f"Invalid contour data in '{CONTOUR_FILE}'. "
            "Please regenerate it by running generate_contours.py."
        )

    return contours


def load_templates() -> Tuple["np.ndarray", "np.ndarray"]:
    """Load the per-square image samples and their labels."""

    import numpy as np

    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES

    def _prepare(samples: "np.ndarray", labels: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
        samples = samples.astype(np.float32, copy=False)
        samples = samples.reshape(samples.shape[0], -1)
        samples = _normalize_rows(samples)
        labels = np.asarray(labels).astype(str)
        return samples, labels

    try:
        with np.load(TEMPLATE_FILE) as data:
            templates = _prepare(data["samples"], data["labels"])
    except FileNotFoundError:
        from piece_templates_data import load_default_templates

        samples, labels = load_default_templates()
        templates = _prepare(samples, labels)
    except Exception as exc:  # pragma: no cover - defensive error handling
        raise RuntimeError(
            f"Unable to load template data from '{TEMPLATE_FILE}'. "
            "Regenerate it with generate_templates.py or remove the file to"
            " use the built-in defaults."
        ) from exc

    _TEMPLATES = templates
    return _TEMPLATES


def infer_castling_rights(board: chess.Board) -> None:
    """Infer and set castling rights based on king and rook positions."""
    rights = []
    if board.piece_at(chess.E1) == chess.Piece.from_symbol("K"):
        if board.piece_at(chess.H1) == chess.Piece.from_symbol("R"):
            rights.append("K")
        if board.piece_at(chess.A1) == chess.Piece.from_symbol("R"):
            rights.append("Q")
    if board.piece_at(chess.E8) == chess.Piece.from_symbol("k"):
        if board.piece_at(chess.H8) == chess.Piece.from_symbol("r"):
            rights.append("k")
        if board.piece_at(chess.A8) == chess.Piece.from_symbol("r"):
            rights.append("q")
    board.set_castling_fen("".join(rights) or "-")


def board_from_image(path: str) -> chess.Board:
    """
    Identifies pieces on a chessboard image using contour matching and returns a chess.Board object.
    """
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image from path: {path}")

    h, w = img.shape[:2]
    square_height = h / 8.0
    square_width = w / 8.0
    square_size = min(square_height, square_width)

    # Use rounded edges so that we cover the full board even if the
    # resolution is not a perfect multiple of eight.
    y_edges = [int(round(rank * square_height)) for rank in range(9)]
    x_edges = [int(round(file * square_width)) for file in range(9)]

    import numpy as np

    samples, labels = load_templates()
    empty_indices = [i for i, label in enumerate(labels) if label == '.']
    empty_templates = samples[empty_indices] if empty_indices else None
    board = chess.Board(None)

    for rank in range(8):
        for file in range(8):
            y0 = y_edges[rank]
            y1 = y_edges[rank + 1]
            x0 = x_edges[file]
            x1 = x_edges[file + 1]

            if y1 <= y0 or x1 <= x0:
                continue

            square_img = img[y0:y1, x0:x1]

            if square_img.size == 0:
                continue

            h_sq, w_sq = square_img.shape
            min_dim = min(h_sq, w_sq)
            border_margin = max(1, int(round(min_dim * 0.05))) if min_dim else 0
            if h_sq > border_margin * 2 and w_sq > border_margin * 2:
                square_img = square_img[border_margin:h_sq - border_margin, border_margin:w_sq - border_margin]

            resized = cv2.resize(
                square_img,
                (CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )

            feature = resized.astype(np.float32).reshape(1, -1)
            feature = _normalize_rows(feature)

            diffs = samples - feature
            errors = np.mean(diffs * diffs, axis=1)
            best_index = int(np.argmin(errors))
            best_symbol = labels[best_index]
            best_error = float(errors[best_index])

            if best_symbol == '.':
                continue

            if empty_templates is not None:
                empty_errors = np.mean((empty_templates - feature) ** 2, axis=1)
                if float(empty_errors.min()) <= best_error * 1.05:
                    continue

            square_index = chess.square(file, 7 - rank)
            board.set_piece_at(square_index, chess.Piece.from_symbol(best_symbol))

    infer_castling_rights(board)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a chessboard screenshot to FEN")
    parser.add_argument("image", help="Path to image containing the board cropped to 8x8 squares")
    args = parser.parse_args()
    board = board_from_image(args.image)
    print(board.fen())


if __name__ == "__main__":
    main()
