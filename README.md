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

Four additional subcommands run a real proving loop with [EZKL](https://github.com/zkonduit/ezkl) 23.0.5, plus a local model registry that gates verification. Input features stay private; only the two circuit outputs are public instances. Nothing is mocked or replaced by digests, and no network is contacted.

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
    --registry verifier/registry.json --model-version v1 \
    --manifest setup/manifest.json \
    --model models/quality.onnx --settings setup/settings.json \
    --vk setup/verification.key --srs setup/srs
```

The verifier additionally keeps a **local model registry** (`model-registry`,
see below) and names an explicitly enabled version on every verification:

```sh
mkdir -p verifier
cp setup/manifest.json setup/settings.json verifier/
cp setup/verification.key verifier/vk
cp setup/srs verifier/
cp models/quality.onnx verifier/model.onnx

# Approve and enable the model version out of band; everything is local/offline.
python -m zkml_quality model-registry register --registry verifier/registry.json \
    --version v1 --manifest verifier/manifest.json --model verifier/model.onnx
python -m zkml_quality model-registry enable --registry verifier/registry.json --version v1

python -m zkml_quality zk-verify --credential credential.json \
    --registry verifier/registry.json --model-version v1 \
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

Any failure — a missing `--registry`, `--model-version`, `--manifest` or
material file, a version that is unregistered, `disabled` or `revoked`, a
corrupt or structurally illegal registry, a mismatch between the enabled
registry record and the trusted manifest, a mismatch between the manifest's
pinned digests and the supplied model/settings/VK/SRS, a credential that
claims a different model or parameters, or proof/instances/public-output
tampering — exits nonzero with a message on stderr and prints no success JSON.

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

Before any cryptographic check, `zk-verify` first requires `--registry` and
`--model-version` to name a version whose registry status is `enabled` —
unregistered, `disabled` and `revoked` versions are refused — and requires the
enabled record to agree with the verifier's own materials (model and manifest
SHA-256, EZKL version, output scale, settings/VK/SRS digests). It then hashes
the verifier's own model, settings, VK and SRS and requires each digest to
equal the value pinned in the verifier's manifest, and validates that the
manifest's version/kind/digests are well-formed. It then cross-checks that the
credential's embedded digests and output scale agree with the manifest
(defence in depth — they can never override the manifest), confirms the
pinned settings encode the same EZKL version and output scale, and finally
runs EZKL verification driven entirely by the manifest-pinned
settings/VK/SRS. The VK was generated for the exact compiled circuit and
settings, so the manifest simultaneously binds the **model** (via
`model_sha256`), the **circuit** (via settings + the matching VK), and the
**verification parameters** (settings/VK/SRS digests), making
model/settings/VK/SRS mixing impossible. A proof made under another circuit or
key cannot pass against the pinned VK even if every credential field is forged
to echo a different trust chain; likewise no credential field can change the
registry decision, which is made entirely from verifier-owned files.

### Model registry: local, offline admission control

`model-registry` manages a single verifier-owned JSON file (`--registry`); it
never contacts the network and never modifies any other file. Versions are
caller-chosen tokens matching `[A-Za-z0-9._-]+` (e.g. `v1`, `2026.09.1`).

```sh
python -m zkml_quality model-registry register \
    --registry verifier/registry.json --version v1 \
    --manifest verifier/manifest.json --model verifier/model.onnx
python -m zkml_quality model-registry list   --registry verifier/registry.json
python -m zkml_quality model-registry enable --registry verifier/registry.json --version v1
python -m zkml_quality model-registry revoke --registry verifier/registry.json --version v1
```

* **register** validates a manifest/model pair the caller approves: the
  manifest must be a well-formed `zk-quality-setup` manifest produced by the
  installed EZKL version, and the ONNX file must hash to the manifest's
  `model_sha256`. It atomically stores the model and manifest SHA-256, EZKL
  version, `output_scale` and settings/VK/SRS digests, initially as
  `disabled`. The registry file (and parent directories) are created if
  missing. Registering the same version with byte-identical pinned content is
  a no-op and leaves the current status untouched; any differing content is a
  conflict and is rejected without overwriting.
* **list** prints one JSON object with a `models` array ordered by version in
  lexicographic order.
* **enable** moves a version from `disabled` to `enabled`; enabling an
  already-enabled version is a no-op.
* **revoke** moves any existing version to `revoked`, permanently. Revocation
  is irreversible: a revoked version can never be enabled or re-registered
  into a different state, and revoking it again is a no-op.

Every successful update is atomic (temp file + rename); a missing, corrupt or
structurally illegal registry is rejected and the existing file is preserved
byte-for-byte. Registry JSON shape:

```json
{
  "format_version": 1,
  "kind": "model-registry",
  "models": {
    "v1": {
      "status": "enabled",
      "model_sha256": "…",
      "manifest_sha256": "…",
      "ezkl_version": "23.0.5",
      "output_scale": 13,
      "artifacts": {"settings": "…", "vk": "…", "srs": "…"}
    }
  }
}
```

The registry belongs to the verifier and is the **admission** layer (which
approved model versions may be checked at all); the independently obtained
manifest remains the **root of trust** for the cryptographic material. The
two are compared on every verification, so a stale or altered registry record
cannot point verification at different settings/VK/SRS than the manifest
pins.

### Proof tasks: an offline prover-side task queue

`proof-task` adds a durable prover-owned task library (`--store`, a local
directory) on top of `zk-prove`, so proving work can be created, executed,
inspected and retried as discrete units. Everything stays local and offline.

```sh
# Record a queued task. Takes exactly the same inputs as zk-prove and
# validates them up front; the task id is allocated atomically.
python -m zkml_quality proof-task create --store tasks \
    --input samples/normal.json --model models/quality.onnx \
    --setup-dir setup --credential credential.json \
    --idempotency-key batch-42

python -m zkml_quality proof-task status --store tasks --task-id pt-…
python -m zkml_quality proof-task run    --store tasks --task-id pt-…
python -m zkml_quality proof-task retry  --store tasks --task-id pt-…
```

* **create** validates the input JSON, model, setup directory and credential
  target exactly as `zk-prove` would and atomically records a `queued` task.
  With `--idempotency-key`, resubmitting the *identical* request returns the
  original task; the same key with a different request is a conflict and is
  rejected.
* **run** performs the only executor transition: `queued` → `running` →
  `succeeded`/`failed`. Each run increments `attempt` and records the
  creation/start/finish timestamps; a per-task executor lock admits a single
  executor, so a competing `run` fails. Success always means a real EZKL
  proof was generated and self-verified, and only then is the credential
  SHA-256 recorded. A failure or interruption never reports success, never
  overwrites a credential and never drops attempt history.
* **status** prints the current task without modifying the store.
* **retry** re-queues a `failed` task whose recorded error is retryable
  (`artifact_missing`, `ezkl_failure`, `output_io`, `interrupted`); every
  other state — and non-retryable failures such as `digest_mismatch` — is
  refused.

Every successful command prints a single JSON object to stdout with
`task_id`, `state`, `attempt`, the timestamps and `credential_sha256`;
failed tasks additionally carry `error.code`, `error.retryable` and a
sanitised `error.message`. Output never contains features, input content,
proof data or filesystem paths. Failures exit nonzero and write only stderr.
Error codes distinguish `invalid_request`, `artifact_missing`,
`digest_mismatch`, `ezkl_failure`, `output_io` and `interrupted`.

The store holds one JSON record per task plus an idempotency index. Records
are strictly validated on every load; a corrupt store or an illegal state is
rejected and the original file is left byte-for-byte intact. All updates go
through a same-directory temporary file and an atomic rename.


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

