from pathlib import Path

import cv2

from fen_converter.board import BoardExtractionError, detect_board, extract_square_images


def test_detect_board(tmp_path: Path) -> None:
    image = Path("examples/diagram.png")
    detected = detect_board(image)
    assert detected.image.shape[0] == detected.image.shape[1]
    squares = extract_square_images(detected.image)
    assert len(squares) == 64
    assert squares["a8"].image.shape[0] == squares["h1"].image.shape[0]


def test_detect_board_failure(tmp_path: Path) -> None:
    bad = tmp_path / "empty.png"
    import numpy as np

    blank = np.zeros((100, 100, 3), dtype="uint8")
    cv2.imwrite(str(bad), blank)
    try:
        detect_board(bad)
    except BoardExtractionError:
        pass
    else:
        raise AssertionError("BoardExtractionError was not raised for a blank image")
