"""Convert a chessboard screenshot to FEN notation."""

import argparse
from typing import Dict

import cv2
import numpy as np
import chess
try:
    import cairosvg
except (OSError, ImportError) as e:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "cairosvg requires the Cairo C library. Install it via your system package "
        "manager (e.g. 'brew install cairo' on macOS or 'apt-get install libcairo2' "
        "on Debian/Ubuntu)"
    ) from e
import chess.svg


PIECE_SYMBOLS = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


def generate_templates(square_size: int) -> Dict[str, np.ndarray]:
    """Generate piece image templates scaled to the given square size."""
    templates = {}
    for symbol in PIECE_SYMBOLS:
        svg = f'<svg width="{square_size}" height="{square_size}" viewBox="0 0 45 45">{chess.svg.PIECES[symbol]}</svg>'
        png_bytes = cairosvg.svg2png(bytestring=svg.encode())
        image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        templates[symbol] = image
    return templates


def board_from_image(path: str) -> chess.Board:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    square_size = min(h // 8, w // 8)
    templates = generate_templates(square_size)

    board = chess.Board(None)
    for rank in range(8):
        for file in range(8):
            y0 = rank * square_size
            x0 = file * square_size
            square_img = img[y0:y0 + square_size, x0:x0 + square_size]
            best_symbol = None
            best_score = 0.0
            for symbol, templ in templates.items():
                if square_img.shape[0] < templ.shape[0] or square_img.shape[1] < templ.shape[1]:
                    continue
                res = cv2.matchTemplate(square_img, templ, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score = score
                    best_symbol = symbol
            if best_symbol and best_score > 0.7:
                square = chess.square(file, 7 - rank)
                board.set_piece_at(square, chess.Piece.from_symbol(best_symbol))
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a chessboard screenshot to FEN")
    parser.add_argument("image", help="Path to image containing the board cropped to 8x8 squares")
    args = parser.parse_args()
    board = board_from_image(args.image)
    print(board.fen())


if __name__ == "__main__":
    main()
