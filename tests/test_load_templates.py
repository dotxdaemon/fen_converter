import numpy as np
from pathlib import Path

import convert


def test_load_templates_uses_module_directory(monkeypatch, tmp_path):
    module_dir = Path(convert.__file__).resolve().parent
    template_path = module_dir / "piece_templates_test.npz"

    samples = np.zeros(
        (2, convert.CLASSIFIER_IMAGE_SIZE, convert.CLASSIFIER_IMAGE_SIZE),
        dtype=np.uint8,
    )
    labels = np.array(["p", "."], dtype="U1")
    np.savez(template_path, samples=samples, labels=labels)

    monkeypatch.setattr(convert, "TEMPLATE_FILE", template_path)
    monkeypatch.setattr(convert, "_TEMPLATES", None)
    monkeypatch.chdir(tmp_path)

    try:
        loaded_samples, loaded_labels = convert.load_templates()
    finally:
        if template_path.exists():
            template_path.unlink()

    assert loaded_samples.shape[0] == 2
    assert loaded_samples.shape[1] == convert.CLASSIFIER_IMAGE_SIZE ** 2
    assert np.allclose(loaded_samples, 0.0)
    assert loaded_labels.tolist() == ["p", "."]
    assert loaded_labels.dtype.kind == "U"
