import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from app import model, preprocess_image


root = Path(__file__).resolve().parent
files = sorted((root / "HOSPITAL").glob("hospital_*/test/*/*"))
files = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
assert files, "No test images found"
matrices = {name: np.zeros((2, 2), dtype=int) for name in ("previous", "training_matched")}

for start in range(0, len(files), 32):
    batch = files[start:start + 32]
    previous, matched, labels = [], [], []
    for path in batch:
        assert path.parent.name in {"NORMAL", "PNEUMONIA"}, path
        with Image.open(path) as img:
            previous.append(np.asarray(img.convert("RGB").resize((100, 100))) / 99.0)
            matched.append(preprocess_image(img)[0])
        labels.append(int(path.parent.name == "PNEUMONIA"))
    for name, pixels in (("previous", previous), ("training_matched", matched)):
        probabilities = model(tf.convert_to_tensor(np.asarray(pixels), dtype=tf.float32), training=False).numpy()
        assert probabilities.shape == (len(batch), 2)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        np.add.at(matrices[name], (labels, probabilities.argmax(axis=1)), 1)
    print(f"Checked {start + len(batch)}/{len(files)} images", flush=True)

report = {
    "images": len(files),
    "classes": ["NORMAL", "PNEUMONIA"],
    "confusion_matrix_axes": "rows=actual, columns=predicted",
    "results": {
        name: {"accuracy": float(matrix.trace() / matrix.sum()), "confusion_matrix": matrix.tolist()}
        for name, matrix in matrices.items()
    },
}
assert all(matrix.sum() == len(files) for matrix in matrices.values())
(root / "prediction_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
