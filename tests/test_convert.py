import sys
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import convert
from convert import infer_castling_rights


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


def test_default_templates_have_two_knights(monkeypatch, tmp_path):
    """Bundled templates should contain exactly two knights per side."""
    monkeypatch.setattr(convert, "TEMPLATE_FILE", str(tmp_path / "missing.npz"))
    monkeypatch.setattr(convert, "_TEMPLATES", None)

    _, labels = convert.load_templates()
    assert np.count_nonzero(labels == "N") == 2
    assert np.count_nonzero(labels == "n") == 2


def test_board_from_image_matches_provided_screenshot(monkeypatch):
    """The shared screenshot should convert to the expected FEN string."""
    monkeypatch.setattr(convert, "_TEMPLATES", None)
    board = convert.board_from_image(str(ROOT / "board.png"))
    assert board.fen() == "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
