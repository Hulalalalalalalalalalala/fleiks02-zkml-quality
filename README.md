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

## Zero-knowledge proofs (EZKL 23.0.5, CPU, offline)

Three additional subcommands run a real proving loop with [EZKL](https://github.com/zkonduit/ezkl) 23.0.5. Input features stay private; only the two circuit outputs are public instances. Nothing is mocked or replaced by digests, and no network is contacted.

```sh
# 1. Compile the circuit and generate settings, SRS, proving and verification keys.
python -m zkml_quality zk-setup --model models/quality.onnx --dir setup
# Writes setup/{compiled.ezkl,settings.json,proving.key,verification.key,srs,manifest.json}.
# The directory must not pre-exist with content; partial output is removed on failure.

# 2. Prover: witness + real proof for one private six-feature input.
python -m zkml_quality zk-prove --input samples/normal.json \
    --model models/quality.onnx --setup-dir setup --credential credential.json

# 3. Verifier: needs only the credential, the zk-setup manifest, and
#    verifier-chosen model/settings/vk/srs. No original input, no proving key,
#    no compiled circuit, no network.
python -m zkml_quality zk-verify --credential credential.json \
    --manifest setup/manifest.json \
    --model models/quality.onnx --settings setup/settings.json \
    --vk setup/verification.key --srs setup/srs
```

On success `zk-verify` prints a single JSON object to stdout containing
`verified`, the two **quantized scores** (integer field instances and their
fixed-point decoding), the `label`, and `model_sha256`. The quantized scores
are what the circuit actually proves (output scale 13 for this model, i.e.
fixed-point = quantized / 8192); they are not the ordinary ONNX floating-point
scores from `infer`, which are never presented as proven. Labels follow the
same rule as `infer`: the larger score wins, a tie is `normal`.

Any tampering or mismatch — proof bytes, public instances, claimed scores or
label, credential model digest, verifier model, settings, VK, SRS, or the
manifest itself — exits nonzero with a message on stderr and prints no success
result.

### Trust boundary

`zk-verify` treats the `--manifest` file (the `manifest.json` written
atomically by `zk-setup`) as the trust root. The verifier must obtain it
independently of the credential — e.g. from the setup operator over a
channel the verifier already trusts. Before any proof is checked, the
manifest is validated (format version, kind, EZKL version, digest shapes) and
the verifier's own model, settings, VK and SRS files are hashed and required
to match it exactly. Only then are the credential's embedded digests compared
against this manifest-bound material, and the final decision still comes from
EZKL cryptographic verification. The credential's claims are never a root of
trust: rewriting `model_sha256` or `verification_artifacts` in the credential,
swapping in a different ONNX model, or mixing verification material from
another setup all fail, because none of that can alter the verifier's
manifest.

### Credential format

`credential.json` is the only artifact exchanged between prover and verifier:

| Field | Content |
| --- | --- |
| `format_version` | Credential format version (currently `1`) |
| `kind` | `zk-quality-credential` |
| `ezkl_version` | EZKL version that produced the proof (`23.0.5`) |
| `model_sha256` | SHA-256 of the ONNX model the circuit was compiled from |
| `verification_artifacts` | SHA-256 digests of `settings.json`, the VK and the SRS |
| `public_output` | `output_scale`, proven `quantized_scores`, fixed-point decoding, and `label` |
| `proof` | The EZKL proof blob including its public instances |

The credential never contains `features`, the input file, or any input path.
The verifier never trusts the credential's embedded model or parameters: they
are accepted only insofar as they agree with the verifier's own manifest-bound
material (see *Trust boundary* above), and the result is accepted only after
EZKL cryptographic verification succeeds against those files.

