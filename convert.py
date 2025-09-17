"""Convert a chessboard screenshot to FEN notation."""

import argparse
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, TYPE_CHECKING, Tuple

import chess

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    import numpy as np

CONTOUR_FILE = "contours.dat"
REQUIRED_CONTOURS = ["P", "N", "B", "R", "Q", "K"]
TEMPLATE_FILE = "piece_templates.npz"
CLASSIFIER_IMAGE_SIZE = 64

_TEMPLATES: Tuple["np.ndarray", "np.ndarray"] | None = None


@dataclass(frozen=True)
class SquarePrediction:
    """Stores the classification candidates for a board square."""

    square: chess.Square
    candidates: Tuple[Tuple[str, float], ...]

    def error_for(self, symbol: str) -> float:
        """Return the matching error for a particular piece symbol."""

        for candidate_symbol, error in self.candidates:
            if candidate_symbol == symbol:
                return error
        return float("inf")

    def best_alternative(self, forbidden: Iterable[str] = ()) -> Tuple[str, float]:
        """Return the lowest-error candidate not in ``forbidden``."""

        forbidden_set = set(forbidden)
        for candidate_symbol, error in self.candidates:
            if candidate_symbol in forbidden_set:
                continue
            return candidate_symbol, error
        return ".", float("inf")

    def best_piece(self) -> Tuple[str | None, float]:
        """Return the lowest-error non-empty candidate for the square."""

        for symbol, error in self.candidates:
            if symbol != ".":
                return symbol, error
        return None, float("inf")


def _replace_with_best_alternative(
    board: chess.Board, prediction: SquarePrediction, forbidden: Iterable[str] = ()
) -> bool:
    """Reassign a square to the best alternative piece that is not forbidden."""

    symbol, _ = prediction.best_alternative(forbidden)
    if symbol == ".":
        board.remove_piece_at(prediction.square)
    else:
        board.set_piece_at(prediction.square, chess.Piece.from_symbol(symbol))
    return True


def _force_piece(
    board: chess.Board, predictions: Dict[chess.Square, SquarePrediction], symbol: str
) -> bool:
    """Ensure that a specific piece symbol is present on the board."""

    piece = chess.Piece.from_symbol(symbol)
    current_square = board.king(piece.color)
    if current_square is not None:
        return False

    best_square: chess.Square | None = None
    best_error = float("inf")

    for square, prediction in predictions.items():
        occupant = board.piece_at(square)
        if occupant is not None and occupant.symbol() in {"K", "k"} and occupant.symbol() != symbol:
            continue
        err = prediction.error_for(symbol)
        if err < best_error:
            best_error = err
            best_square = square

    if best_square is None or best_error == float("inf"):
        # Fall back to the square whose best candidate has the lowest error.
        for square, prediction in predictions.items():
            if not prediction.candidates:
                continue
            occupant = board.piece_at(square)
            if occupant is not None and occupant.symbol() in {"K", "k"} and occupant.symbol() != symbol:
                continue
            candidate_symbol, candidate_error = prediction.candidates[0]
            if candidate_symbol in {"K", "k"} and candidate_symbol != symbol:
                continue
            if candidate_error < best_error:
                best_error = candidate_error
                best_square = square

    if best_square is None:
        return False

    board.set_piece_at(best_square, piece)
    return True


def _prune_extra_kings(
    board: chess.Board, predictions: Dict[chess.Square, SquarePrediction]
) -> bool:
    """Remove surplus kings, keeping the most confident ones."""

    changed = False
    for color, symbol in ((chess.WHITE, "K"), (chess.BLACK, "k")):
        kings = list(board.pieces(chess.KING, color))
        if len(kings) <= 1:
            continue

        keep_square = min(
            kings,
            key=lambda sq: predictions.get(sq, SquarePrediction(sq, tuple())).error_for(symbol),
        )

        for square in kings:
            if square == keep_square:
                continue
            prediction = predictions.get(square)
            if prediction is None:
                board.remove_piece_at(square)
            else:
                _replace_with_best_alternative(
                    board, prediction, forbidden={symbol, "K", "k"}
                )
            changed = True

    return changed


def _remove_worst_piece(
    board: chess.Board,
    predictions: Dict[chess.Square, SquarePrediction],
    *,
    color: chess.Color | None = None,
    symbol: str | None = None,
) -> bool:
    """Downgrade the piece with the highest matching error for the given filter."""

    worst_square: chess.Square | None = None
    worst_error = -1.0

    for square, prediction in predictions.items():
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_symbol = piece.symbol()
        if symbol is not None and piece_symbol != symbol:
            continue
        if color is not None and piece.color != color:
            continue
        if piece_symbol in {"K", "k"} and symbol not in {"K", "k"}:
            continue

        error = prediction.error_for(piece_symbol)
        if error == float("inf") and prediction.candidates:
            error = prediction.candidates[-1][1]

        if error > worst_error:
            worst_error = error
            worst_square = square

    if worst_square is None:
        return False

    piece_symbol = board.piece_at(worst_square).symbol()  # type: ignore[union-attr]
    forbidden = {piece_symbol}
    if piece_symbol in {"K", "k"}:
        forbidden.update({"K", "k"})

    prediction = predictions.get(worst_square)
    if prediction is None:
        board.remove_piece_at(worst_square)
        return True

    return _replace_with_best_alternative(board, prediction, forbidden)


def _fix_backrank_pawns(
    board: chess.Board, predictions: Dict[chess.Square, SquarePrediction]
) -> bool:
    """Convert pawns on the first or eighth rank to plausible alternatives."""

    changed = False
    for color, pawn_symbol in ((chess.WHITE, "P"), (chess.BLACK, "p")):
        for square in list(board.pieces(chess.PAWN, color)):
            rank = chess.square_rank(square)
            if rank not in (0, 7):
                continue

            prediction = predictions.get(square)
            if prediction is None:
                board.remove_piece_at(square)
            else:
                _replace_with_best_alternative(board, prediction, forbidden={pawn_symbol})
            changed = True

    return changed


def _resolve_check_conflicts(
    board: chess.Board, predictions: Dict[chess.Square, SquarePrediction]
) -> bool:
    """Address impossible check scenarios by adjusting low-confidence pieces."""

    checker_squares = list(board.checkers())
    if not checker_squares:
        return False

    def _score(square: chess.Square) -> float:
        piece = board.piece_at(square)
        if piece is None:
            return float("inf")
        prediction = predictions.get(square)
        if prediction is None:
            return float("inf")
        return prediction.error_for(piece.symbol())

    worst_square = max(checker_squares, key=_score)
    prediction = predictions.get(worst_square)
    if prediction is None:
        board.remove_piece_at(worst_square)
        return True

    symbol = board.piece_at(worst_square).symbol()  # type: ignore[union-attr]
    forbidden = {symbol}
    if symbol in {"K", "k"}:
        forbidden.update({"K", "k"})

    _replace_with_best_alternative(board, prediction, forbidden)
    return True


def _add_high_confidence_pieces(
    board: chess.Board,
    predictions: Dict[chess.Square, SquarePrediction],
    *,
    min_improvement: float = 0.0,
) -> bool:
    """Insert pieces on empty squares that strongly prefer non-empty candidates."""

    candidates: list[tuple[float, chess.Square, str]] = []
    for square, prediction in predictions.items():
        if board.piece_at(square) is not None:
            continue

        symbol, piece_error = prediction.best_piece()
        if symbol is None:
            continue

        empty_error = prediction.error_for(".")
        if empty_error == float("inf"):
            improvement = float("inf")
        else:
            improvement = empty_error - piece_error

        candidates.append((improvement, square, symbol))

    candidates.sort(reverse=True, key=lambda item: item[0])

    changed = False
    for improvement, square, symbol in candidates:
        if improvement <= min_improvement:
            break

        if symbol in {"K", "k"}:
            king_square = board.king(chess.WHITE if symbol == "K" else chess.BLACK)
            if king_square is not None:
                continue

        board.set_piece_at(square, chess.Piece.from_symbol(symbol))
        if board.status() != chess.Status.VALID:
            board.remove_piece_at(square)
            continue

        changed = True

    return changed


def ensure_legal_board(
    board: chess.Board, predictions: Dict[chess.Square, SquarePrediction]
) -> None:
    """Mutate ``board`` so that it represents a legal chess position."""

    for _ in range(128):
        status = board.status()
        if status == chess.Status.VALID:
            break

        if status & chess.Status.BAD_CASTLING_RIGHTS:
            infer_castling_rights(board)
            continue

        if status & chess.Status.INVALID_EP_SQUARE:
            board.ep_square = None
            continue

        if status & chess.Status.NO_WHITE_KING:
            if _force_piece(board, predictions, "K"):
                continue

        if status & chess.Status.NO_BLACK_KING:
            if _force_piece(board, predictions, "k"):
                continue

        if status & chess.Status.TOO_MANY_KINGS:
            if _prune_extra_kings(board, predictions):
                continue

        if status & chess.Status.TOO_MANY_WHITE_PAWNS:
            if _remove_worst_piece(board, predictions, symbol="P"):
                continue

        if status & chess.Status.TOO_MANY_BLACK_PAWNS:
            if _remove_worst_piece(board, predictions, symbol="p"):
                continue

        if status & chess.Status.PAWNS_ON_BACKRANK:
            if _fix_backrank_pawns(board, predictions):
                continue

        if status & chess.Status.TOO_MANY_WHITE_PIECES:
            if _remove_worst_piece(board, predictions, color=chess.WHITE):
                continue

        if status & chess.Status.TOO_MANY_BLACK_PIECES:
            if _remove_worst_piece(board, predictions, color=chess.BLACK):
                continue

        if status & (
            chess.Status.OPPOSITE_CHECK
            | chess.Status.TOO_MANY_CHECKERS
            | chess.Status.IMPOSSIBLE_CHECK
        ):
            if _resolve_check_conflicts(board, predictions):
                continue

        if status & chess.Status.EMPTY:
            changed = False
            changed |= _force_piece(board, predictions, "K")
            changed |= _force_piece(board, predictions, "k")
            if changed:
                continue

        if not _remove_worst_piece(board, predictions):
            break

    infer_castling_rights(board)

    if board.status() != chess.Status.VALID:
        raise RuntimeError(
            "Unable to infer a legal chess position from the provided image"
        )


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
    label_to_indices: Dict[str, "np.ndarray"] = {}
    for symbol in np.unique(labels):
        label_to_indices[str(symbol)] = np.flatnonzero(labels == symbol)

    board = chess.Board(None)
    predictions: Dict[chess.Square, SquarePrediction] = {}

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

            symbol_errors = []
            for symbol, indices in label_to_indices.items():
                if indices.size == 0:
                    continue
                symbol_errors.append((symbol, float(errors[indices].min())))

            square_index = chess.square(file, 7 - rank)
            if not symbol_errors:
                predictions[square_index] = SquarePrediction(square_index, tuple())
                continue

            symbol_errors.sort(key=lambda item: item[1])
            prediction = SquarePrediction(square_index, tuple(symbol_errors))
            predictions[square_index] = prediction

            best_symbol, best_error = prediction.candidates[0]
            empty_error = prediction.error_for('.')

            if best_symbol != '.':
                board.set_piece_at(square_index, chess.Piece.from_symbol(best_symbol))

    ensure_legal_board(board, predictions)
    if _add_high_confidence_pieces(board, predictions):
        ensure_legal_board(board, predictions)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a chessboard screenshot to FEN")
    parser.add_argument("image", help="Path to image containing the board cropped to 8x8 squares")
    args = parser.parse_args()
    board = board_from_image(args.image)
    print(board.fen())


if __name__ == "__main__":
    main()
