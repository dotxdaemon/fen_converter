import pickle
from pathlib import Path

import numpy as np
import pytest

import convert


def test_load_contours_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "missing.dat"
    monkeypatch.setattr(convert, "CONTOUR_FILE", str(missing))
    with pytest.raises(RuntimeError):
        convert.load_contours()


def test_load_contours_corrupted_file(monkeypatch, tmp_path):
    corrupted = tmp_path / "contours.dat"
    corrupted.write_bytes(b"not a pickle")
    monkeypatch.setattr(convert, "CONTOUR_FILE", str(corrupted))
    with pytest.raises(RuntimeError):
        convert.load_contours()


def test_load_contours_invalid_data(monkeypatch, tmp_path):
    invalid = tmp_path / "contours.dat"
    # Save an object that is not a dict or missing keys
    invalid_data = {"P": np.array([1, 2, 3])}
    with open(invalid, "wb") as f:
        pickle.dump(invalid_data, f)
    monkeypatch.setattr(convert, "CONTOUR_FILE", str(invalid))
    with pytest.raises(RuntimeError):
        convert.load_contours()


def test_load_contours_uses_module_directory(monkeypatch, tmp_path):
    module_dir = Path(convert.__file__).resolve().parent
    contour_path = module_dir / "contours_test.dat"
    contour_data = {
        symbol: np.full((2, 2), idx, dtype=np.uint8)
        for idx, symbol in enumerate(convert.REQUIRED_CONTOURS)
    }

    with open(contour_path, "wb") as f:
        pickle.dump(contour_data, f)

    monkeypatch.setattr(convert, "CONTOUR_FILE", contour_path)
    monkeypatch.chdir(tmp_path)

    try:
        loaded = convert.load_contours()
    finally:
        if contour_path.exists():
            contour_path.unlink()

    assert set(loaded) == set(contour_data)
    for key, expected in contour_data.items():
        assert np.array_equal(loaded[key], expected)
