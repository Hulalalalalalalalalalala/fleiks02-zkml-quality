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

## Zero-knowledge proofs (EZKL 23.0.5, CPU)

`zk-setup`, `zk-prove`, and `zk-verify` add a real EZKL proof loop over the same six-feature contract. The features are a **private** witness; only the two output scores are public. Nothing is simulated and no ONNX-runtime score is placed in a credential.

```sh
# One-time ceremony per model: compiled circuit, settings, SRS, proving and verifying keys.
python -m zkml_quality zk-setup --model models/quality.onnx --dir outputs/setup

# Prover: needs the model, the setup directory (including pk.key), and the private input.
python -m zkml_quality zk-prove \
  --input samples/inspect.json --model models/quality.onnx \
  --setup-dir outputs/setup --credential outputs/inspect.credential.json

# Verifier: needs ONLY the credential and its own model, settings, vk and SRS.
# No original input, no proving key, and no network access.
python -m zkml_quality zk-verify \
  --credential outputs/inspect.credential.json --model models/quality.onnx \
  --settings outputs/setup/settings.json --vk outputs/setup/vk.key --srs outputs/setup/kzg.srs
```

On success `zk-verify` prints exactly one JSON object to stdout, for example:

```json
{"verified": true, "quantized_scores": [-1.8125, 2.8125], "label": "inspect", "model_sha256": "6d23c4ec6e0cdb838b0e2662621ec2932647828b413a022f07ff1fd18f6d2561"}
```

### Proven outputs versus `infer` outputs

The circuit works in fixed point (output scale 2^13). The credential's `quantized_scores` are the circuit's **actual public outputs**, reconstructed from the verified proof's field elements, and `label` is the maximum of those two quantized values, with ties selecting `normal`. They can differ in the last binary digits from the ordinary `infer` float scores; only the quantized values are claimed as proven. `infer` and `demo` keep their existing float behavior unchanged.

### Credential format

A credential is a single JSON object with exactly these top-level fields:

- `format_version` — credential schema version (`1.0`).
- `ezkl_version` — the pinned EZKL release (`23.0.5`).
- `model_sha256` — SHA-256 of the ONNX model whose weights are proven.
- `artifacts` — SHA-256 digests of the compiled circuit, settings, verifying key, and SRS.
- `proof` — the real EZKL proof JSON, base64-encoded.
- `public_output` — `output_scale`, the two public field-element `instances`, the decoded `quantized_integers`, `quantized_scores`, and the `label`.

Credentials never contain the raw `features`, input file contents, or input paths. During verification the embedded model hash and artifact digests are only compared against the **verifier's own** files; the credential's `public_output` summary is not trusted — it must match the proof's instances exactly, and the proof must verify against the verifier-supplied settings, vk, and SRS.

### Failure handling

Any invalid path, JSON, six-feature input, or EZKL artifact; any model/settings/vk/SRS mismatch; any tampered proof, claim, or artifact; and any EZKL failure prints a clear `error: ...` message to stderr and exits nonzero with no success result on stdout. Setup writes into a private staging area and only publishes all artifacts together, so mismatched intermediate files are never reused.
