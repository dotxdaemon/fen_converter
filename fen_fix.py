"""
fen_fix.py - An alternative FEN converter for screenshots.

This script is a drop‑in replacement for the original `convert.py` from the
`fen_converter` project.  The upstream converter assumes that the input image
already contains a cropped 8×8 chessboard in the standard orientation
with white on the bottom and that the `python‑chess` library is
available.  In practice screenshots often include borders, move arrows,
coordinate annotations and may even be flipped horizontally or vertically.
Additionally, the execution environment used in this challenge does not
permit installation of arbitrary Python packages, so we cannot rely on
`python‑chess` being present.

This module performs the following steps:

1.  **Load the screenshot** in colour and convert it to grayscale for
    processing.
2.  **Detect the bounding box of the board** by thresholding out the
    dark background.  The board is assumed to be the largest nearly
    square region of non‑background pixels.  We crop a square region
    from either the right or the bottom of this bounding box depending
    on which side is larger.  A small margin (5 % of the side length)
    is trimmed from all four sides to remove the light grey frame
    typically surrounding the board.
3.  **Generate four candidate orientations** by optionally flipping
    the cropped board horizontally and/or vertically.  The original
    screenshot shows the board with black at the bottom and the files
    reversed, so simply passing the raw crop to the classifier yields
    nonsensical output.  We therefore evaluate all four orientations
    and pick the one with the lowest template matching error.
4.  **Classify each square** using the built‑in templates from
    `piece_templates_data.py`.  Each square is resized to 64×64
    pixels, normalised to zero mean and unit variance and then compared
    against every template using mean squared error.  The template
    whose error is smallest is selected as the prediction for that
    square.  The total error across all squares is used to choose the
    best orientation.
5.  **Construct a FEN string** directly without relying on
    `python‑chess`.  Once we have an 8×8 array of piece symbols we
    generate the piece placement part of the FEN by collapsing runs of
    empty squares into digits.  Castling rights are inferred by
    checking whether kings and rooks appear on their starting squares
    (e1/e8 and a1/h1/a8/h8) and are still present.  Because the side
    to move and en‑passant target square are unknowable from a static
    picture we default them to "w" (white to move) and "-".  The
    halfmove clock and fullmove number are set to 0 and 1 respectively
    as placeholders.

Example usage:

    python fen_fix.py --image path/to/screenshot.png

The script prints the resulting FEN string to standard output.

This implementation is intentionally self contained: it imports only
`cv2` and `numpy` for image processing and uses the baked in templates
from the original project.  It does not depend on `python‑chess`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

import os
import sys

# Ensure we can import the bundled template loader from the original
# repository regardless of the current working directory.  The
# built‑in templates shipped with fen_converter were generated for a
# Lichess style board and will not match Chess.com screenshots.  Users
# can supply their own template archive via the ``--templates`` option.
REPO_DIR = os.path.join(os.path.dirname(__file__), "fen_converter-main")
if REPO_DIR not in sys.path and os.path.isdir(REPO_DIR):
    sys.path.insert(0, REPO_DIR)
from piece_templates_data import load_default_templates


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalise a flattened image to zero mean and unit variance.

    Each template and test patch is normalised prior to computing the
    mean squared error.  Without this normalisation the matching
    becomes sensitive to lighting differences across the board.  This
    behaviour mirrors the logic used in the upstream project.
    """

    vec = vec.astype(np.float32, copy=False)
    mean = vec.mean()
    std = vec.std()
    # Avoid division by zero for completely uniform images
    if std < 1e-6:
        std = 1.0
    return (vec - mean) / std


@dataclass
class SquareClassification:
    """Stores the predicted symbol and error for a board square."""

    symbol: str
    error: float


def _crop_board(gray: np.ndarray) -> np.ndarray:
    """Extract a square image of the chessboard from the screenshot.

    There are two independent heuristics for extracting the board
    region:

    1.  **Full image crop**: take the minimum of the height and width
        of the image and crop a square of that size from the right (if
        width > height) or from the bottom (if height > width).  This
        works for screenshots where the board occupies the right or
        bottom portion of the frame and is bordered by dark UI
        elements.

    2.  **Mask‑based crop**: threshold the image to identify light
        pixels, take the bounding box of all non‑background pixels and
        crop a square from that bounding box.  This helps when the
        surrounding UI includes bright elements.

    Both crops are attempted and the one whose side length is largest
    is chosen.  A 5 % margin is trimmed from all four sides to remove
    the grey frame around the board.
    """

    h, w = gray.shape
    # First candidate: square anchored to the right or bottom based on full image size
    board_size_full = min(h, w)
    if w >= h:
        # Crop from the right
        full_board = gray[0:board_size_full, w - board_size_full : w]
    else:
        # Crop from the bottom
        full_board = gray[h - board_size_full : h, 0:board_size_full]
    # Second candidate: mask‑based crop
    # Use a low threshold to exclude the dark background; the board is
    # significantly lighter.  Empirically 20 suffices for this
    # screenshot.
    mask = gray > 20
    coords = np.column_stack(np.where(mask))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        h_mask = y1 - y0 + 1
        w_mask = x1 - x0 + 1
        if h_mask <= w_mask:
            board_size_mask = h_mask
            crop_y0 = y0
            crop_y1 = y0 + board_size_mask
            crop_x1 = x1
            crop_x0 = crop_x1 - board_size_mask
        else:
            board_size_mask = w_mask
            crop_x0 = x0
            crop_x1 = x0 + board_size_mask
            crop_y1 = y1
            crop_y0 = crop_y1 - board_size_mask
        mask_board = gray[crop_y0:crop_y1, crop_x0:crop_x1]
    else:
        mask_board = None
        board_size_mask = -1
    # Choose the candidate with the larger side length.  This helps to
    # avoid degenerate crops when the mask is incomplete.
    if board_size_mask > board_size_full:
        board = mask_board
        board_size = board_size_mask
    else:
        board = full_board
        board_size = board_size_full
    # Trim 5 % of the side length from each side to remove the light
    # frame around the squares.  Use at least one pixel per side.
    margin = max(1, int(round(board_size * 0.05)))
    if board.shape[0] > margin * 2 and board.shape[1] > margin * 2:
        board = board[margin:-margin, margin:-margin]
    return board


def _classify_board(board: np.ndarray, template_features: np.ndarray, template_labels: List[str]) -> Tuple[List[List[SquareClassification]], float]:
    """Classify all 64 squares of the board and return total error.

    Parameters
    ----------
    board: np.ndarray
        The cropped square grayscale image of the board.  It will be
        subdivided into an 8×8 grid.  Squares need not be perfectly
        aligned – rounding is used to split the image as evenly as
        possible.
    template_features: np.ndarray
        An array of shape (n_templates, 4096) containing normalised
        flattened template images.
    template_labels: List[str]
        The corresponding symbol for each template.

    Returns
    -------
    List[List[SquareClassification]]
        An 8×8 nested list of classification results.  The outer list
        corresponds to ranks (rows) from top (rank 8) to bottom
        (rank 1).  Each inner list contains `SquareClassification`
        instances ordered from file a to file h.
    float
        The sum of matching errors across all squares.  This value is
        used to pick the best board orientation.
    """

    h, w = board.shape
    # Generate split positions.  Round the boundaries so that small
    # cumulative errors do not propagate across multiple squares.
    y_edges = [int(round(i * h / 8.0)) for i in range(9)]
    x_edges = [int(round(i * w / 8.0)) for i in range(9)]
    total_error = 0.0
    classifications: List[List[SquareClassification]] = []
    for rank in range(8):  # 0 is top rank (rank 8)
        row_classifications: List[SquareClassification] = []
        for file in range(8):  # 0 is file a
            y0 = y_edges[rank]
            y1 = y_edges[rank + 1]
            x0 = x_edges[file]
            x1 = x_edges[file + 1]
            if y1 <= y0 or x1 <= x0:
                # Skip degenerate squares
                row_classifications.append(SquareClassification(symbol='.', error=float('inf')))
                continue
            square_img = board[y0:y1, x0:x1]
            # Trim a small border inside each square to avoid edges and arrows
            h_sq, w_sq = square_img.shape
            border = max(1, int(round(min(h_sq, w_sq) * 0.05)))
            if h_sq > border * 2 and w_sq > border * 2:
                square_img = square_img[border:-border, border:-border]
            # Resize to the classifier size
            resized = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)
            feature = _normalize(resized.reshape(-1))
            # Compute mean squared error to each template
            diffs = template_features - feature  # broadcasting
            errors = np.mean(diffs * diffs, axis=1)
            best_idx = int(np.argmin(errors))
            best_symbol = template_labels[best_idx]
            best_error = float(errors[best_idx])
            row_classifications.append(SquareClassification(symbol=best_symbol, error=best_error))
            total_error += best_error
        classifications.append(row_classifications)
    return classifications, total_error


def infer_castling_rights(board: List[List[str]]) -> str:
    """Infer castling rights from a board representation.

    Parameters
    ----------
    board: List[List[str]]
        8×8 matrix of piece symbols.  The first index is rank (0 is
        rank 8/top), the second index is file (0 is file a).  Symbols
        use the standard Forsyth–Edwards notation: uppercase for white,
        lowercase for black and '.' for empty.

    Returns
    -------
    str
        The castling rights portion of a FEN string.  Possible values
        include any combination of "KQkq" or "-" if none are
        available.
    """

    rights: List[str] = []
    # White king at e1 (file e is index 4 on rank 1 which is board[7])
    if board[7][4] == 'K':
        if board[7][7] == 'R':
            rights.append('K')
        if board[7][0] == 'R':
            rights.append('Q')
    # Black king at e8 (rank 8 is board[0])
    if board[0][4] == 'k':
        if board[0][7] == 'r':
            rights.append('k')
        if board[0][0] == 'r':
            rights.append('q')
    return ''.join(rights) or '-'


def board_to_fen(board: List[List[str]]) -> str:
    """Convert a board representation into a FEN string.

    Only the piece placement, side to move, castling rights, en
    passant target, halfmove and fullmove counters are included.  We
    default the side to move to white and the latter two counters to
    0 and 1 since they cannot be inferred from a static image.
    """

    # Piece placement: ranks are separated by '/'; within a rank
    # consecutive empty squares are replaced by a digit.
    fen_rows: List[str] = []
    for rank in range(8):
        row = board[rank]
        run = 0
        fen_row = ''
        for piece in row:
            if piece == '.':
                run += 1
            else:
                if run > 0:
                    fen_row += str(run)
                    run = 0
                fen_row += piece
        if run > 0:
            fen_row += str(run)
        fen_rows.append(fen_row)
    placement = '/'.join(fen_rows)
    castling = infer_castling_rights(board)
    # Always assume it's white's move and no en passant available
    return f"{placement} w {castling} - 0 1"


def orientations(board: np.ndarray) -> Iterable[np.ndarray]:
    """Yield the four possible board orientations to test.

    The original board may be flipped horizontally, vertically or both.
    Yielding them in a deterministic order ensures reproducibility.  The
    order returned is: original, horizontal flip, vertical flip, both
    flips.
    """

    yield board
    yield cv2.flip(board, 1)  # horizontal flip
    yield cv2.flip(board, 0)  # vertical flip
    yield cv2.flip(board, -1)  # both flips


def convert_image_to_fen(path: str, *, templates: str | None = None) -> str:
    """Convert a screenshot of a chessboard into a FEN string.

    Parameters
    ----------
    path : str
        Path to the image file.
    templates : str or None, optional
        Path to a ``.npz`` file containing precomputed piece templates
        and their labels.  If not supplied, the bundled default
        templates are used.  For Chess.com style boards the default
        templates (built for Lichess) perform poorly; supplying a
        custom template archive generated from a labelled board of the
        same style will improve accuracy dramatically.

    Returns
    -------
    str
        A full FEN string including piece placement, side to move,
        castling rights, en passant square (always ``-``), halfmove
        clock and fullmove number.

    This function performs the following operations:
        * Loads the image and converts it to grayscale.
        * Detects the board boundaries by finding strong vertical and
          horizontal edges using a Gaussian‑smoothed gradient.  The
          outermost left/right and top/bottom edges with large
          gradients are assumed to correspond to the board frame.
        * Crops a square region around the board and trims a small
          margin to remove the grey border.
        * Evaluates all four possible orientations (original,
          horizontal flip, vertical flip and both flips) using the
          specified templates and selects the orientation with the
          lowest total matching error.
        * Constructs a FEN string from the resulting 8×8 grid of
          predicted piece symbols.  Castling rights are inferred from
          the presence of kings and rooks on their starting squares.
    """

    # Load the image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image from path: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Detect board boundaries using vertical and horizontal gradients
    # Compute absolute differences between neighbouring pixels along x and y
    vert_diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
    vert_score = vert_diff.sum(axis=0)
    vert_score_smooth = cv2.GaussianBlur(vert_score.reshape(1, -1), (1, 51), 0).flatten()
    # Threshold at 50 % of the maximum to find edge columns
    thresh_x = vert_score_smooth.max() * 0.5
    x_edges = np.where(vert_score_smooth > thresh_x)[0]
    if x_edges.size < 2:
        raise RuntimeError("Unable to detect vertical board edges")
    # Cluster adjacent edge indices to group into left and right edges
    import itertools
    clusters_x: List[Tuple[int, int]] = []
    last = -1000
    start = end = None
    for idx in x_edges:
        if start is None:
            start = end = idx
        elif idx <= end + 10:
            end = idx
        else:
            clusters_x.append((start, end))
            start = end = idx
    if start is not None:
        clusters_x.append((start, end))
    # Choose the leftmost and rightmost clusters
    board_x0, _ = clusters_x[0]
    _, board_x1 = clusters_x[-1]
    # Repeat for horizontal edges
    horiz_diff = np.abs(np.diff(gray.astype(np.float32), axis=0))
    horiz_score = horiz_diff.sum(axis=1)
    horiz_score_smooth = cv2.GaussianBlur(horiz_score.reshape(-1, 1), (51, 1), 0).flatten()
    thresh_y = horiz_score_smooth.max() * 0.5
    y_edges = np.where(horiz_score_smooth > thresh_y)[0]
    if y_edges.size < 2:
        raise RuntimeError("Unable to detect horizontal board edges")
    clusters_y: List[Tuple[int, int]] = []
    last = -1000
    start = end = None
    for idx in y_edges:
        if start is None:
            start = end = idx
        elif idx <= end + 10:
            end = idx
        else:
            clusters_y.append((start, end))
            start = end = idx
    if start is not None:
        clusters_y.append((start, end))
    board_y0, _ = clusters_y[0]
    _, board_y1 = clusters_y[-1]
    # Crop the board rectangle
    board_rect = gray[board_y0:board_y1, board_x0:board_x1]
    # Ensure a square crop by taking the minimum dimension and anchoring
    # at the top‑left corner of the rectangle.  This discards any
    # excess on the right or bottom which may arise from asymmetric
    # borders.
    rect_h, rect_w = board_rect.shape
    side = int(min(rect_h, rect_w))
    square = board_rect[0:side, 0:side].copy()
    # Trim a 5 % margin to remove the grey border
    margin = max(1, int(round(side * 0.05)))
    if square.shape[0] > margin * 2 and square.shape[1] > margin * 2:
        board_img = square[margin:-margin, margin:-margin]
    else:
        board_img = square
    # Load templates: if a custom archive is provided use it, otherwise
    # fall back to the bundled defaults
    if templates:
        data = np.load(templates)
        samples = data["samples"]
        labels = data["labels"].astype(str)
    else:
        samples, labels = load_default_templates()
    n_templates = samples.shape[0]
    template_features = np.empty((n_templates, 64 * 64), dtype=np.float32)
    for i in range(n_templates):
        template_features[i] = _normalize(samples[i].reshape(-1))
    template_labels = [str(l) for l in labels]
    # Evaluate all orientations and pick the one with minimum error
    best_error = float("inf")
    best_board_symbols: List[List[str]] | None = None
    for oriented in orientations(board_img):
        classifications, total_err = _classify_board(oriented, template_features, template_labels)
        if total_err < best_error:
            best_error = total_err
            best_board_symbols = [[cls.symbol for cls in row] for row in classifications]
    if best_board_symbols is None:
        raise RuntimeError("Classification failed for all orientations")
    return board_to_fen(best_board_symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a chessboard screenshot to FEN")
    parser.add_argument("--image", required=True, help="Path to the screenshot image")
    parser.add_argument(
        "--templates",
        default=None,
        help=(
            "Optional path to a .npz file containing custom piece templates. "
            "Providing templates generated from a board image of the same style "
            "(e.g. Chess.com) greatly improves accuracy over the default Lichess templates."
        ),
    )
    args = parser.parse_args()
    fen = convert_image_to_fen(args.image, templates=args.templates)
    print(fen)


if __name__ == "__main__":
    main()