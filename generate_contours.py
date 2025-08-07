import pickle
import re
from typing import Dict, List

import cv2
import numpy as np
import chess
import chess.svg
try:
    import cairosvg
except (OSError, ImportError) as e:
    raise RuntimeError(
        "cairosvg requires the Cairo C library. Install it via your system package "
        "manager (e.g. 'brew install cairo' on macOS or 'apt-get install libcairo2' "
        "on Debian/Ubuntu)"
    ) from e

PIECE_SYMBOLS = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
# Note: The shapes of white and black pieces are the same (e.g., P and p).
# We only need to generate contours for one set of pieces.
UNIQUE_PIECE_TYPES = ["P", "N", "B", "R", "Q", "K"]
CONTOUR_FILE = "contours.dat"
IMAGE_SIZE = 200  # Use a larger size for higher resolution contours

def generate_contours() -> Dict[str, np.ndarray]:
    """
    Generates template contours for each unique piece type and saves them to a file.
    """
    contours = {}
    print("Generating piece contours...")
    for symbol in UNIQUE_PIECE_TYPES:
        # Generate a black piece on a white background SVG
        piece_guts = chess.svg.PIECES[symbol]
        cleaned_guts = re.sub(r'\s(fill|stroke|stroke-width)="[^"]*"', '', piece_guts)
        styled_guts = cleaned_guts.replace('<g ', '<g fill="black" ', 1)

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{IMAGE_SIZE}" '
            f'height="{IMAGE_SIZE}" viewBox="0 0 45 45" '
            f'style="background-color:white">{styled_guts}</svg>'
        )

        png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))

        # Decode and convert to grayscale
        image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        # Threshold to get a binary image. The piece is black (0), so we want to find it.
        # THRESH_BINARY_INV makes the piece white (255) and background black (0).
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        found_contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not found_contours:
            raise RuntimeError(f"Could not find contour for piece '{symbol}'")

        # Find the largest contour by area, which should be the piece
        largest_contour = max(found_contours, key=cv2.contourArea)
        contours[symbol] = largest_contour
        print(f"  - Generated contour for {symbol}")

    # Save the contours to a file
    with open(CONTOUR_FILE, "wb") as f:
        pickle.dump(contours, f)
    print(f"\nContours saved to {CONTOUR_FILE}")

    return contours

if __name__ == "__main__":
    generate_contours()
