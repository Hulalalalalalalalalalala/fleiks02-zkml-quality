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

# 3. Verifier: pin model/circuit/parameters with the verifier-owned manifest.
#    The verifier independently obtains manifest.json from a zk-setup run they
#    trust (e.g. their own) and the four public files it names. No original
#    input, no proving key, no compiled circuit, no network.
python -m zkml_quality zk-verify --credential credential.json \
    --manifest setup/manifest.json \
    --model models/quality.onnx --settings setup/settings.json \
    --vk setup/verification.key --srs setup/srs
```

For example, the verifier can keep their trust material separate from the
prover's (they never receive `proving.key` or `compiled.ezkl`):

```sh
mkdir -p verifier
cp setup/manifest.json setup/settings.json setup/verification.key setup/srs verifier/
cp models/quality.onnx verifier/model.onnx
python -m zkml_quality zk-verify --credential credential.json \
    --manifest verifier/manifest.json --model verifier/model.onnx \
    --settings verifier/settings.json --vk verifier/vk --srs verifier/srs
```

On success `zk-verify` prints a single JSON object to stdout containing
`verified`, the two **quantized scores** (integer field instances and their
fixed-point decoding), the `label`, and `model_sha256` — nothing else, and no
file paths. The quantized scores are what the circuit actually proves (output
scale 13 for this model, i.e. fixed-point = quantized / 8192); they are not
the ordinary ONNX floating-point scores from `infer`, which are never
presented as proven. Labels follow the same rule as `infer`: the larger score
wins, a tie is `normal`.

Any failure — a missing `--manifest` or material file, an illegal/malformed
manifest field or digest, a mismatch between the manifest's pinned digests and
the supplied model/settings/VK/SRS, a credential that claims a different model
or parameters, or proof/instances/public-output tampering — exits nonzero with
a message on stderr and prints no success JSON.

### Trust boundary: the verifier manifest is the root of trust

`manifest.json` is produced atomically together with the keys by `zk-setup`.
It pins:

| Manifest field | Pins |
| --- | --- |
| `format_version` / `kind` | Format (`1`) and setup kind (`zk-quality-setup`) |
| `ezkl_version` | The exact EZKL version that generated everything (`23.0.5`) |
| `model_sha256` | SHA-256 of the ONNX model |
| `artifacts.settings` | SHA-256 of `settings.json` (circuit shape, scales, visibility) |
| `artifacts.vk` / `artifacts.srs` | SHA-256 of the verification key and SRS |
| (`artifacts.compiled` / `artifacts.pk`) | Also pinned for the prover; not needed to verify |

The verifier must obtain `manifest.json` **independently** (typically by running
`zk-setup` themselves, or over a channel they trust). It must **not** come from
the prover, from inside the credential, or be regenerated/overwritten from
credential claims — a prover who controls the root of trust could otherwise
rewrite the model digest and verification-parameter digests while reusing the
old proof.

Before any cryptographic check, `zk-verify` first hashes the verifier's own
model, settings, VK and SRS and requires each digest to equal the value pinned
in the verifier's manifest, and validates that the manifest's version/kind/
digests are well-formed. It then cross-checks that the credential's embedded
digests and output scale agree with the manifest (defence in depth — they can
never override the manifest), confirms the pinned settings encode the same
EZKL version and output scale, and finally runs EZKL verification driven
entirely by the manifest-pinned settings/VK/SRS. The VK was generated for the
exact compiled circuit and settings, so the manifest simultaneously binds the
**model** (via `model_sha256`), the **circuit** (via settings + the matching
VK), and the **verification parameters** (settings/VK/SRS digests), making
model/settings/VK/SRS mixing impossible. A proof made under another circuit or
key cannot pass against the pinned VK even if every credential field is forged
to echo a different trust chain.

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
Its embedded model and parameter digests are **not** a root of trust: they are
only cross-checked against the verifier's own manifest. The real anchor is
`--manifest`, which independently pins the model, circuit (via settings and the
matching VK) and verification parameters; acceptance requires every supplied
file to match the manifest and EZKL cryptographic verification to succeed.

