from fen_converter.fen import build_fen
from fen_converter.labeler import SquareLabel, SquareSuggestion


def make_label(square: str, symbol: str) -> SquareLabel:
    suggestion = SquareSuggestion(square=square, symbol=symbol, confidence=1.0, reason="test")
    return SquareLabel(square=square, symbol=symbol, suggestion=suggestion)


def test_build_fen_full() -> None:
    labels = [
        make_label("a8", "r"),
        make_label("h8", "r"),
        make_label("e1", "K"),
        make_label("h1", "R"),
    ]
    fen = build_fen(labels)
    assert fen.startswith("r6r/8/8/8/8/8/8/4K2R")


def test_build_fen_empty() -> None:
    fen = build_fen([])
    assert fen.startswith("8/8/8/8/8/8/8/8")
