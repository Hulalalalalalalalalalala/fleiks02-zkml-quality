# Quality inference

Small local equipment-quality inference example using a public ONNX network and synthetic sensor features. The illustrative weights are not a validated industrial detector.

```sh
python -m pip install -r requirements.txt
python -m zkml_quality demo
python -m zkml_quality infer --input samples/normal.json
python -m unittest discover -s tests
```

Input JSON contains `features`: exactly six finite numbers in [0, 1], ordered as vibration, temperature, acoustic noise, current, pressure deviation, and wear. Outputs contain two raw scores, the maximum-score label (`normal` or `inspect`), and the SHA-256 of the ONNX file. Ties select `normal`. Scores are not probabilities. Invalid input exits nonzero without a successful result.

The current interface runs ordinary local inference and a two-sample demo. It does not authenticate sensor provenance. `python tools/export_model.py` reproduces the checked-in example model from its public weights.
