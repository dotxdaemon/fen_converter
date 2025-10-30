from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np

from .board import SquareImage

_FILES = "abcdefgh"
_RANKS = "87654321"
_VALID_SYMBOLS = ".kqrbnpKQRBNP"


def _square_order() -> List[str]:
    return [f"{file}{rank}" for rank in _RANKS for file in _FILES]


@dataclasses.dataclass
class SquareSuggestion:
    square: str
    symbol: str
    confidence: float
    reason: str


@dataclasses.dataclass
class SquareLabel:
    square: str
    symbol: str
    suggestion: SquareSuggestion

    def is_empty(self) -> bool:
        return self.symbol == "."


class SquareClassifier:
    def __init__(self, empty_threshold: float = 18.0) -> None:
        self.empty_threshold = empty_threshold

    def suggest(self, square: SquareImage) -> SquareSuggestion:
        grayscale = np.mean(square.image[..., :3], axis=2)
        interior = grayscale[4:-4, 4:-4]
        stddev = float(interior.std())
        mean = float(interior.mean())
        occupied = stddev >= self.empty_threshold
        if not occupied:
            return SquareSuggestion(square.square, ".", 1.0, "low texture")

        symbol = "P" if mean > 150 else "p"
        reason = f"occupied (std={stddev:.1f}) brightness={mean:.0f}"
        confidence = min(stddev / 255.0 + 0.1, 0.99)
        return SquareSuggestion(square.square, symbol, confidence, reason)


def label_squares(
    squares: Dict[str, SquareImage],
    *,
    classifier: Optional[SquareClassifier] = None,
    interactive: bool = True,
) -> List[SquareLabel]:
    classifier = classifier or SquareClassifier()
    order = _square_order()
    labels: List[Optional[SquareLabel]] = [None] * len(order)
    index = 0

    while index < len(order):
        square_name = order[index]
        square = squares[square_name]
        suggestion = classifier.suggest(square)
        symbol = suggestion.symbol

        if interactive:
            ascii_art = square.as_ascii()
            print("\n" + "=" * 40)
            print(f"Square {square_name}")
            print(ascii_art)
            print(f"Suggestion: {suggestion.symbol!r} ({suggestion.reason})")
            entered = input(
                "Enter piece (KQRBNP for white, kqrbnp for black, '.' empty, 'back' undo, 'q' quit): "
            ).strip()
            lower_entered = entered.lower()
            if entered in _VALID_SYMBOLS:
                symbol = entered
            elif lower_entered == "q":
                raise KeyboardInterrupt("Labeling aborted by user")
            elif lower_entered in {"back", "undo", "u"}:
                if index > 0:
                    index -= 1
                continue
            elif entered:
                print(f"Invalid symbol {entered!r}, using suggestion {symbol!r}.")

        labels[index] = SquareLabel(square=square_name, symbol=symbol, suggestion=suggestion)
        index += 1

    return [label for label in labels if label is not None]
