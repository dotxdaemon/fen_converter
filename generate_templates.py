"""Generate piece templates from a labelled board image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import chess

TEMPLATE_FILE = Path("piece_templates.npz")
IMAGE_SIZE = 64
BORDER_FRACTION = 0.05


def preprocess_square(square_img: np.ndarray) -> np.ndarray:
    if square_img.size == 0:
        raise ValueError("Empty square image")

    h_sq, w_sq = square_img.shape
    min_dim = min(h_sq, w_sq)
    border = max(1, int(round(min_dim * BORDER_FRACTION)))
    if h_sq > border * 2 and w_sq > border * 2:
        square_img = square_img[border:h_sq - border, border:w_sq - border]

    resized = cv2.resize(square_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def collect_templates(image_path: Path, fen: str) -> Tuple[List[np.ndarray], List[str]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    board = chess.Board(fen)
    h, w = img.shape
    square_height = h / 8.0
    square_width = w / 8.0
    y_edges = [int(round(r * square_height)) for r in range(9)]
    x_edges = [int(round(f * square_width)) for f in range(9)]

    samples: List[np.ndarray] = []
    labels: List[str] = []

    for rank in range(8):
        for file in range(8):
            y0, y1 = y_edges[rank], y_edges[rank + 1]
            x0, x1 = x_edges[file], x_edges[file + 1]
            square_img = img[y0:y1, x0:x1]
            if square_img.size == 0:
                continue

            square_index = chess.square(file, 7 - rank)
            piece = board.piece_at(square_index)
            key = piece.symbol() if piece else '.'
            samples.append(preprocess_square(square_img))
            labels.append(key)

    return samples, labels


def save_templates(samples: List[np.ndarray], labels: List[str], output: Path) -> None:
    np.savez_compressed(
        output,
        samples=np.stack(samples, axis=0),
        labels=np.array(labels, dtype="<U1"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate piece templates from a labelled board image")
    parser.add_argument("image", type=Path, help="Path to the board image")
    parser.add_argument("fen", type=str, help="FEN describing the board")
    parser.add_argument(
        "--output",
        type=Path,
        default=TEMPLATE_FILE,
        help="Output file (default: piece_templates.npz)",
    )
    args = parser.parse_args()

    samples, labels = collect_templates(args.image, args.fen)
    save_templates(samples, labels, args.output)
    print(f"Templates saved to {args.output}")


if __name__ == "__main__":
    main()
