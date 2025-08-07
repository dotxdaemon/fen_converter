"""Convert a chessboard screenshot to FEN notation."""

import argparse
import pickle
from typing import Dict

import cv2
import numpy as np
import chess
import chess.svg

CONTOUR_FILE = "contours.dat"
PIECE_SYMBOLS = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


def load_contours() -> Dict[str, np.ndarray]:
    """Load the pre-generated piece contours from a file."""
    try:
        with open(CONTOUR_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"Contour file '{CONTOUR_FILE}' not found. "
            "Please run generate_contours.py first to create it."
        )


def infer_castling_rights(board: chess.Board) -> None:
    """Infer and set castling rights based on king and rook positions."""
    # This is a naive implementation and might not be correct for all positions,
    # but it's what was in the original script.
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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image from path: {path}")

    h, w = img.shape[:2]
    square_size = min(h // 8, w // 8)

    template_contours = load_contours()
    board = chess.Board(None)

    for rank in range(8):
        for file in range(8):
            y0 = rank * square_size
            x0 = file * square_size
            # Add a small buffer to avoid cutting off pieces at the edge
            y1 = y0 + square_size
            x1 = x0 + square_size
            square_img = img[y0:y1, x0:x1]

            # Use adaptive thresholding to handle different square colors
            # ADAPTIVE_THRESH_GAUSSIAN_C is often good for variable lighting
            # Block size must be odd
            block_size = max(square_size // 4, 5)
            if block_size % 2 == 0:
                block_size += 1

            binary_square = cv2.adaptiveThreshold(
                square_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, block_size, 5
            )

            contours, _ = cv2.findContours(binary_square, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            # Filter out very small contours (noise) and find the most likely piece contour
            best_match_score = float('inf')
            best_match_symbol = None
            largest_contour = None

            # Find the largest contour, assuming it's the piece
            found_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if not found_contours:
                continue

            main_contour = found_contours[0]

            # Ignore contours that are too small or too large to be a piece
            area = cv2.contourArea(main_contour)
            if area < (square_size * 0.1)**2 or area > (square_size * 0.9)**2:
                continue

            for piece_symbol, template_contour in template_contours.items():
                # Compare the contour shape with the templates
                score = cv2.matchShapes(main_contour, template_contour, cv2.CONTOURS_MATCH_I2, 0.0)
                if score < best_match_score:
                    best_match_score = score
                    best_match_symbol = piece_symbol

            # If the best match is good enough, determine the piece color
            if best_match_score < 0.35:  # Relaxed threshold a bit
                # Improved color detection: compare piece brightness to its background
                piece_mask = np.zeros(square_img.shape, np.uint8)
                cv2.drawContours(piece_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
                piece_mean = cv2.mean(square_img, mask=piece_mask)[0]

                # Invert mask to get the background
                bg_mask = cv2.bitwise_not(piece_mask)
                bg_mean = cv2.mean(square_img, mask=bg_mask)[0]

                # If piece is lighter than its background, it's white.
                is_white = piece_mean > bg_mean

                final_symbol = best_match_symbol.upper() if is_white else best_match_symbol.lower()

                square_index = chess.square(file, 7 - rank)
                board.set_piece_at(square_index, chess.Piece.from_symbol(final_symbol))

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
