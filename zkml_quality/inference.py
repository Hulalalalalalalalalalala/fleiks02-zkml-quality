import hashlib
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "quality.onnx"


def validate_features(document):
    """Validate the README's six-feature contract; return the ordered feature list.

    Features are, in order: vibration, temperature, acoustic noise, current,
    pressure deviation, and wear. Each must be a finite number in [0, 1].
    """
    if not isinstance(document, dict):
        raise ValueError("Input must be a JSON object")
    features = document.get("features")
    if not isinstance(features, list) or len(features) != 6:
        raise ValueError("features must contain exactly six numbers")
    for value in features:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("features must be finite numbers in [0, 1]")
    return list(features)


def infer(document):
    features = validate_features(document)
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    output = session.run(["scores"], {"features": np.asarray([features], dtype=np.float32)})[0][0]
    return {"scores": output.tolist(), "label": ("normal", "inspect")[int(np.argmax(output))],
            "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest()}
