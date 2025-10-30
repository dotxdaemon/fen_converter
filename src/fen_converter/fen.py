from __future__ import annotations

from typing import Iterable, List

from .labeler import SquareLabel

_FILES = "abcdefgh"
_RANKS = "87654321"


def build_fen(labels: Iterable[SquareLabel], active_color: str = "w", castling: str = "-", en_passant: str = "-", halfmove: int = 0, fullmove: int = 1) -> str:
    squares = {label.square: label.symbol for label in labels}
    ranks: List[str] = []
    for rank in _RANKS:
        empties = 0
        fen_rank = []
        for file in _FILES:
            symbol = squares.get(f"{file}{rank}", ".")
            if symbol == ".":
                empties += 1
            else:
                if empties:
                    fen_rank.append(str(empties))
                    empties = 0
                fen_rank.append(symbol)
        if empties:
            fen_rank.append(str(empties))
        ranks.append("".join(fen_rank))
    placement = "/".join(ranks)
    return f"{placement} {active_color} {castling} {en_passant} {halfmove} {fullmove}"
