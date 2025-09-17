import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import convert
from convert import (
    SquarePrediction,
    ensure_legal_board,
    infer_castling_rights,
)


def test_infer_castling_rights_start_position():
    """Castling rights should be inferred for starting position pieces."""
    start = chess.Board()
    board = chess.Board(None)
    for square, piece in start.piece_map().items():
        board.set_piece_at(square, piece)
    infer_castling_rights(board)
    assert board.fen() == chess.STARTING_FEN


def test_normalize_rows_handles_zero_variance():
    """Normalization should not introduce NaNs when the input is constant."""
    vec = np.zeros((1, 4), dtype=np.float32)
    normalized = convert._normalize_rows(vec)
    assert normalized.shape == vec.shape
    assert np.allclose(normalized, 0.0)


def test_load_templates_uses_embedded_defaults(monkeypatch, tmp_path):
    """The converter should fall back to bundled template data."""
    monkeypatch.setattr(convert, "TEMPLATE_FILE", str(tmp_path / "missing.npz"))
    monkeypatch.setattr(convert, "_TEMPLATES", None)
    samples, labels = convert.load_templates()
    assert samples.shape[0] == labels.shape[0] > 0
    assert samples.ndim == 2


def test_load_templates_decodes_byte_labels(monkeypatch, tmp_path):
    """Byte-encoded labels should be decoded to strings for classification."""
    path = tmp_path / "custom_templates.npz"
    samples = np.zeros((2, convert.CLASSIFIER_IMAGE_SIZE, convert.CLASSIFIER_IMAGE_SIZE), dtype=np.uint8)
    labels = np.array([b"p", b"."], dtype="S1")
    np.savez(path, samples=samples, labels=labels)

    monkeypatch.setattr(convert, "TEMPLATE_FILE", str(path))
    monkeypatch.setattr(convert, "_TEMPLATES", None)

    _, loaded_labels = convert.load_templates()
    assert loaded_labels.dtype.kind == "U"
    assert isinstance(loaded_labels[0], str)


def _prediction(square: chess.Square, mapping):
    return SquarePrediction(square, tuple((symbol, float(error)) for symbol, error in mapping))


def test_ensure_legal_board_adds_missing_kings():
    """Missing kings should be restored using the most confident squares."""

    board = chess.Board(None)
    predictions = {
        chess.E1: _prediction(chess.E1, [("K", 0.05), (".", 0.1)]),
        chess.E8: _prediction(chess.E8, [("k", 0.02), (".", 0.2)]),
    }

    ensure_legal_board(board, predictions)

    assert board.king(chess.WHITE) == chess.E1
    assert board.king(chess.BLACK) == chess.E8


def test_ensure_legal_board_removes_backrank_pawns():
    """Backrank pawns should be replaced with alternative candidates."""

    board = chess.Board(None)
    board.set_piece_at(chess.A1, chess.Piece.from_symbol("P"))
    board.set_piece_at(chess.A8, chess.Piece.from_symbol("p"))

    predictions = {
        chess.A1: _prediction(chess.A1, [("P", 0.5), (".", 0.05)]),
        chess.A8: _prediction(chess.A8, [("p", 0.5), (".", 0.1)]),
        chess.E1: _prediction(chess.E1, [("K", 0.01), (".", 0.2)]),
        chess.E8: _prediction(chess.E8, [("k", 0.01), (".", 0.2)]),
    }

    ensure_legal_board(board, predictions)

    assert board.piece_at(chess.A1) is None
    assert board.piece_at(chess.A8) is None
    assert board.king(chess.WHITE) == chess.E1
    assert board.king(chess.BLACK) == chess.E8


def test_high_confidence_additions_restore_pieces():
    """Squares that heavily favour a piece should gain that piece."""

    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece.from_symbol("K"))
    board.set_piece_at(chess.E8, chess.Piece.from_symbol("k"))

    predictions = {
        chess.E1: _prediction(chess.E1, [("K", 0.01), (".", 0.4)]),
        chess.E8: _prediction(chess.E8, [("k", 0.02), (".", 0.5)]),
        chess.E4: _prediction(chess.E4, [(".", 0.9), ("B", 0.1), ("N", 0.3)]),
        chess.D4: _prediction(chess.D4, [(".", 0.15), ("q", 0.35)]),
        chess.C4: _prediction(chess.C4, [(".", 0.31), ("N", 0.30), ("B", 0.5)]),
    }

    added = convert._add_high_confidence_pieces(board, predictions)

    assert added is True
    piece = board.piece_at(chess.E4)
    assert piece is not None and piece.symbol() == "B"
    slight_piece = board.piece_at(chess.C4)
    assert slight_piece is not None and slight_piece.symbol() == "N"
    assert board.piece_at(chess.D4) is None
