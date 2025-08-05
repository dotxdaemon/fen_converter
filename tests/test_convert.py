import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert import infer_castling_rights


def test_infer_castling_rights_start_position():
    """Castling rights should be inferred for starting position pieces."""
    start = chess.Board()
    board = chess.Board(None)
    for square, piece in start.piece_map().items():
        board.set_piece_at(square, piece)
    infer_castling_rights(board)
    assert board.fen() == chess.STARTING_FEN
