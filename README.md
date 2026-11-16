# Disagreement Analysis Pipeline

End-to-end pipeline for discovering **why** annotators disagree. Given a
multi-annotator CSV, it produces a ranked, statistically tested list of
hypotheses about the textual features that drove disagreement.

Two **sample configurations** and their annotation guidelines are bundled.
These are starting points, not ready-to-run experiments: input datasets are
not included, and dataset paths and model connections must be configured
before running.

| Sample configuration | Task | Annotators |
|---|---|---|
| [yelp_v1.yaml](configs/disagreement_configs/yelp/yelp_v1.yaml) | Restaurant review topic classification with food, staff, and time labels | 2 (three labels) |
| [ghc-syn_v1.yaml](configs/disagreement_configs/ghc-syn/ghc-syn_v1.yaml) | Hate-speech labels (Gab Hate Corpus, synthetic LLM annotators) | 2 (binary) |

---

## 1. Pipeline overview

```
                   ┌──────────────────────────────────────────────────────┐
                   │ run_disagreement.sh                                  │
                   │  ─ explicit setup installs hashed dependencies     │
                   │  ─ launches run_disagreement_cli.py                  │
                   └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ run_disagreement_cli.py (Rich CLI)                               │
   │   1. pick project + version                                      │
   │   2. silently scaffold YAML if missing                           │
   │   3. open dashboard (Method | Run | Annotation Eval)             │
   │   4. Save persists only, Run starts execution                    │
   │   5. disagreement_orchestrator.run(cfg)                          │
   │   6. validation eval loop  + evaluate.py                         │
   │   7. held-out eval loop    + evaluate.py                         │
   └──────────────────────────────────────────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
 disagreement_explainer            annotation_orchestrator             evaluate.py
 (preprocess → rationales →        (per-batch judge LLM run            (per-hypothesis χ²/Fisher
 synthesizer → DPMeans clusters    over hypothesis concepts)           + multivariate logit ranking)
 → disagreement_explanations.json)
```

The six logical stages are:

1. **Preprocessing** — load CSV, compute per-row disagreement, stratified
   train/validation/held-out split (60/20/20), pairwise pruning + chunking.
2. **Rationale generation** — for each chunk, the rationales LLM explains
   why each annotator picked their label.
3. **Synthesis** — the synthesizer LLM turns each pair of rationales into a
   single hypothesis statement.
4. **Clustering** — hypotheses are embedded with `all-mpnet-base-v2` and
   clustered with DPMeans; centroid-nearest hypotheses are written to
   `disagreement_explanations.json`.
5. **Validation eval** — top hypotheses are converted to binary judge
   prompts and tested against the validation slice; significance results
   land in `significant_disagreement_explanations.csv`.
6. **Held-out eval** — significant hypotheses are re-tested on the unseen
   held-out slice for honest reporting.

---

## 2. Prerequisites

Use **CPython 3.11** on Apple Silicon with macOS 14 or newer, or Linux x86_64
with glibc 2.28 or newer. These minimums follow the selected PyTorch wheels.
Other Python versions and platforms are outside the locked environment matrix.
Each model stage needs an explicit provider, model, endpoint, and authentication
mode. OpenAI and compatible hosted services use environment referenced keys.
Local services can use explicit authentication free mode on literal loopback
addresses.

Setup is an explicit operation. Ordinary launches verify installed versions
against the selected lock and never install or upgrade packages.

```bash
bash src/run_disagreement.sh --setup
# Optional browser profile
bash src/run_disagreement.sh --setup --browser
```

Setup creates `.venv` and installs the platform lock with pip hash checking.
Failures stop setup. Browser setup additionally downloads Chromium.
Setup and ordinary launches require the four platform lock files under `locks/`.
Regenerate missing locks through the tooling instructions below. Do not disable
verification or hash checking to work around missing files.

The dependency repair updated manifests and locks to PyTorch 2.13.0 and
Requests 2.34.2. Validation used an isolated environment and deliberately left
the existing `.venv` unchanged. Updating files does not patch an existing
installation. Run explicit setup before using the new versions. If Playwright
is already installed, include `--browser` so its profile also matches the lock.

---

## 3. Quickstart — interactive

After setup, launch the wrapper and configure your input CSV and model stages
in the dashboard.

```bash
bash src/run_disagreement.sh
```

The CLI will:

1. Prompt for **project** (e.g. `yelp`) and **version** (`new`, `latest`,
   or a specific number).
2. Open a three-pane dashboard at `http://localhost:<port>`:
   * **Method Setup** — locations, rationales LLM, synthesizer LLM.
   * **Run Setup** — pipeline knobs (`example_size`, `num_threads`, etc.)
     plus a live chunk-count preview.
   * **Annotation (Eval)** — eval model, iterations, batch size.
3. **Save** persists edits and exits without execution. Changed settings create
   a new version. **Run** saves edits and performs preflight before starting
   discovery, validation, and heldout evaluation.

Cancel closes the dashboard and stops the launch without running pipeline
stages. A newly scaffolded configuration is retained. Saving unchanged settings
keeps the existing version. Invalid form values and failed saves leave the
dashboard open with an error.

Save permits incomplete inputs and unavailable credentials. Run requires ready
inputs and credentials only for stages that need live requests. Existing keys
are never displayed or accepted in the dashboard.

The launch URL carries a temporary capability in its fragment. The browser
removes it immediately. Sensitive reads require authorization. Mutations also
require the exact local origin and host. Relaunch rather than bookmarking this
temporary URL. The capability is not written to browser storage or terminal logs.

Preview access is limited to visible CSV datasets and Markdown guidelines
inside the project. Approve an external root explicitly at launch if needed.

```bash
bash src/run_disagreement.sh --allow-preview-root /absolute/path/to/inputs
```

Do not expose the dashboard through a proxy or port forwarding.

The preview uses the selected annotation cap, sampling seed, and debug setting.
After changing them, click Refresh to recompute. The split seed remains fixed
at 42. `train_on_test` reuses data across all splits and must not be used to
claim independent heldout performance.

To skip the dashboard and re-run an already-saved YAML:

```bash
bash src/run_disagreement.sh --skip-setup --project yelp --version 1
```

This runs without a confirmation prompt. Before execution, the CLI checks
local configuration and inputs and shows the selected models, paths, and
stages that will run or reuse existing artifacts. These checks do not contact
the provider or verify that remote credentials are valid.

---

## 4. Batch run — both sample configurations

Use the **batch runner** to run the sample configurations (`yelp`,
`ghc-syn`) in sequence after supplying input CSV paths and model connections.
It uses retry/backoff on transient API failures
(rate limits, 5xx, network blips):

Despite its filename, `run_all_mock.sh` performs real API calls and can incur
charges. Only `CHARDIS_DRY_RUN=1` prints commands without launching them.

```bash
# Both sample configurations, v1 each (after setting inputs and connections)
bash src/disagreement_analysis/run_all_mock.sh

# Subset (positional args)
bash src/disagreement_analysis/run_all_mock.sh yelp

# Multiple versions
VERSIONS="1 2" bash src/disagreement_analysis/run_all_mock.sh

# See what *would* run, without actually launching
CHARDIS_DRY_RUN=1 bash src/disagreement_analysis/run_all_mock.sh

# Validation only (skip Stage 6 held-out)
CHARDIS_SKIP_HELDOUT=1 bash src/disagreement_analysis/run_all_mock.sh
```

Tunable env vars:

| Var | Default | Purpose |
|---|---|---|
| `VERSIONS` | `"1"` | Space-separated version numbers to run |
| `MAX_ATTEMPTS` | `3` | Max attempts per project × version |
| `RETRY_BASE_DELAY` | `60` | Base seconds between attempts (exponential backoff) |
| `PYTHON` | `.venv/bin/python` if present, else `python3` | Python interpreter |
| `CHARDIS_SKIP_HELDOUT` | `0` | Skip the held-out eval stage (validation-only run) |
| `CHARDIS_SKIP_VALIDATION` | unset | Skip validation and use existing validation artifacts for heldout evaluation |
| `CHARDIS_DRY_RUN` | `0` | Print planned commands and exit |

Per-project stdout is teed to `outputs/disagreement_outputs/<project>/v<n>/run_log.txt`.
Discovery, annotation batches, and rankings are reused only when their
completion manifests and content hashes match. Fingerprints include relevant
inputs, prompts, configuration, implementation files, and dependency locks.
Changed or unmanifested artifacts are rejected without overwriting them.
Use a new version for a changed experiment.

Failed annotation batches produce an incomplete status and preserve successful
raw batches for a matching retry. They do not produce a completed final CSV or
permit evaluation. Exclusive writer locks prevent concurrent writes to the
same run directory. Inspect an abandoned `.run.lock` before removing it.

The batch runner does not install dependencies or configure authentication.
Prepare the Python environment, input CSV paths, and all stage credentials
before launching it.

---

## 5. Configuration

Each saved project version is a single YAML file at:

```
configs/disagreement_configs/<project>/<project>_v<n>.yaml
```

Four top-level sections:

| Section | What it controls |
|---|---|
| `rationales` | Rationale model and its explicit connection and request settings |
| `synthesizer` | LLM that combines rationale pairs into single hypotheses. Same shape as `rationales`. |
| `disagreement_explainer` | Pipeline knobs: `data_path`, `guidelines_path`, `output_path`, `num_threads`, `example_size`, `more_than_2`, `train_on_test`, `random_state`. |
| `evaluation` | Independently configured judge connection, `model`, iterations, batching, and hypothesis caps |

New templates and bundled sample configurations leave the input dataset path,
model, and endpoint blank.
Fill these fields for each stage using the existing nested YAML format.

Set `disagreement_explainer.data_path.value` to a full CSV path, such as
`/absolute/path/to/inputs/annotations.csv`, or to its directory and set
`data_filename.value` to the CSV filename. Standalone annotation configurations
use `annotation.data_path` and `annotation.data_filename` instead.

Disagreement input CSVs need a unique `unique_id`, a nonempty `text` column,
and per-annotator label columns. All columns other than `unique_id` and `text`
are interpreted as annotators, so omit unrelated metadata columns. The default
mode expects two annotators; enable `more_than_2` for additional annotators.
Use annotation guidelines appropriate to your labels.

For dashboard previews of external CSVs, authorize their containing directory
with `--allow-preview-root`. Saving an incomplete configuration is allowed,
but running requires a readable input CSV. Version changes preserve your
configured input location.

```yaml
provider:
  value: compatible
auth_mode:
  value: api-key
api_key_env:
  value: MODEL_API_KEY
endpoint:
  value: ''
credential_endpoint:
  value: ''
credential_provider:
  value: compatible
model:
  value: ''
message_role:
  value: system
```

Use these same field names in every model section, including `evaluation`.
Standalone annotation configurations use `llm`. Configure each stage
independently. References can intentionally name the same exported
environment variable. The pipeline does not load `.env` files.

`provider` selects the service adapter. `auth_mode` selects authentication.
`api_key_env` names the environment variable holding the credential. It never
contains the credential itself. `api_version` and `scope` remain accepted as
legacy compatibility fields. Current adapters do not use them for version
routing or OAuth token acquisition.
Credential destination bindings remain separate from the connection endpoint.

Existing names remain accepted on input.

| Legacy name | Public name |
|---|---|
| `openai_model`, `evaluation.eval_model` | `model`, `evaluation.model` |
| `openai_endpoint`, `base_url` | `endpoint` |
| `openai_api_version` | `api_version` |
| `openai_key`, `api_key` | `key`, for legacy migration only |
| `rationales_openai` | `rationales` |
| `synthesizer_openai` | `synthesizer` |
| Annotation section `openai` | `llm` |

Loading a file does not rewrite it. New templates, explicit saves, and generated
judge YAML use the public names while preserving descriptions and values.
Equivalent aliases are accepted. Conflicting values, metadata, or duplicate
sections must be resolved explicitly. New templates omit legacy key fields.

Connection CLI options accept `--model`, `--endpoint`, `--base_url`, and
`--api_version` alongside their old spellings. Evaluation preparation also
accepts `--eval_model` as an alias for `--model`. Conflicting duplicate options
fail rather than silently choosing the last value. The separate results
analysis command keeps its `--eval_model` output folder argument.

Names normalize to the same settings for artifact identity checks. This code
update still changes implementation hashes, so existing manifests may no longer
match. Use a new run or the explicit legacy import procedure where applicable.
Nothing automatically reseals or overwrites historical artifacts.

For compatible hosting, use `provider` set to `compatible` and explicitly bind
the chosen endpoint. For local services, use `provider` set to `local`,
`auth_mode` set to `none`, and a literal address such as
`http://127.0.0.1:8000/v1`. No ambient key is sent in local mode.

The retained hosted adapters are `compatible` and `openai`, both using
`auth_mode` set to `api-key`. Export `MODEL_API_KEY`, or explicitly configure
another environment variable name. Existing supported connections keep their
configured credential references. Configurations selecting unsupported
providers or authentication modes fail and require explicit reconfiguration.
They are never silently routed to another service.

Endpoint or provider changes require editing `credential_endpoint` and
`credential_provider` deliberately in YAML. The dashboard cannot approve this
rebinding. Authenticated requests do not follow redirects or inherit proxy
settings. Choose `system` or `developer` explicitly for the model. Select only
one positive token limit. Unsupported request parameters fail rather than
being silently removed. Negative sentinel settings omit optional generation
parameters according to the existing request contract.

Plaintext keys are rejected on save and execution. To migrate a legacy key,
first provision its exact value in an environment variable, then run the
explicit removal command. Supply a binding for every secret bearing section.

```bash
.venv/bin/python -m src.migrate_credentials path/to/config.yaml \
  --binding rationales=RATIONALE_KEY \
  --binding synthesizer=SYNTHESIS_KEY --confirm-removal
```

Migration verifies the environment copies without printing or backing up keys.
Bindings accept both public and legacy section names. Then set explicit
provider, authentication, and destination bindings.

Complete historical discovery artifacts can be associated with the current
inputs only through an explicit import.

```bash
.venv/bin/python -m src.import_legacy_cache path/to/config.yaml --confirm-association
```

Import checks split identities and content, preserves the original files, and
marks historical model provenance as unknown. Historical judge caches and
rankings are not imported. Use fresh evaluation output directories. Do not
relabel historical results as newly reproducible.

---

## 6. Outputs

Per-version artifacts land under
`outputs/disagreement_outputs/<project>/v<n>/`:

```
v1/
├── disagreement_explanations.json      ← Stage 4 result (ranked hypotheses)
├── metadata.json                       ← corpus + chunk + pruning stats
├── raw/                                ← per-chunk JSON + cluster artefacts
├── train.csv
├── validation/
│   ├── validation.csv
│   └── <eval_model>/
│       ├── eval_dataset.csv
│       ├── hypo_predictiveness.csv
│       ├── overall_predictiveness_logit.csv
│       └── significant_disagreement_explanations.csv
└── heldout/
    ├── test.csv
    └── <eval_model>/
        ├── overall_predictiveness_logit.csv
        └── min/overall_predictiveness_logit.csv
```

The "headline" results are:

* `disagreement_explanations.json` — every candidate hypothesis the
  pipeline discovered, ordered by cluster size.
* `validation/<eval_model>/significant_disagreement_explanations.csv` —
  the subset that passed independence testing on validation.
* `heldout/<eval_model>/overall_predictiveness_logit.csv` contains
  predictive results identified by `features_set` and `threshold`.

With per iteration files, validation uses max aggregation, meaning any
positive judgment is positive, at the model output root. Heldout evaluation
uses the same aggregation there and adds `min/`, where all heldout iterations
must be positive. The min pass still fits on max aggregated validation data.
Without iteration files, evaluation uses the annotator's saved aggregation.

Rows labeled `all (validation set only)` are validation results even inside
the heldout CSV. For the existing overall heldout result, select the
`all_significant` feature set and threshold `0.5`. Other thresholds and feature
sets are separate comparisons. Do not select a headline result merely because
it is the first row or has the highest score.

The CLI lists available and missing output files after execution. Batch run
logs are in `v<n>/run_log.txt`. Interactive runs do not create that log.

Unknown pricing or missing cached token usage is reported as unknown rather
than zero cost. The embedding model is pinned to revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. Its first production use may
download model files. Pinning does not imply bitwise identical remote model
responses or model behavior across hardware.

Dataset text, generated rationales, and generated hypothesis descriptions
travel as structured untrusted user content. Trusted prompts define the output
contract. Responses with invalid identities, types, or shapes fail validation.
These controls reduce instruction confusion but cannot guarantee that a model
ignores every adversarial input.

## Research alignment

These workflow fixes preserve the current scientific behavior. The supplied
paper specifies screening at 0.05, while the evaluation entry point uses 0.1.
This difference remains unresolved. Current outputs must not be described as
using the paper's 0.05 threshold.

The paper also describes adjustable method thresholds and particular model
settings. This copy fixes DPMeans delta at 0.3 and uses entropy 0.9 and overlap
50 in preprocessing. The bundled sample configurations and new templates require explicit
model selection and do not claim to reproduce historical model deployments.
The sample configurations alone do not reproduce the paper's full six dataset
experiment suite; the original experiment inputs must be supplied separately.

---

## 7. Going deeper

Regenerate locks only during an intentional environment update. The resolver
uses the configured pip index and retains SHA256 hashes for pip installation.
Optional profiles are constrained by their platform base lock.
Keep tooling separate from the runtime environment.

```bash
python3.11 -m venv .venv-tools
.venv-tools/bin/python -m pip install -r requirements-tools.txt
.venv-tools/bin/python -m src.lock_dependencies
PYTHONDONTWRITEBYTECODE=1 .venv-tools/bin/python -m unittest discover -s tests -p 'test_lock_dependencies.py' -v
```

The configured mirror failed to serve wheel metadata for tokenizers 0.20.3.
The restored locks were generated through that mirror using the source archive
for resolver metadata instead. Hash checked wheel installation remains enabled.
Tokenizers 0.20.3 is explicitly constrained to avoid the yanked 0.20.4 release.
If the same metadata error occurs, use the resolver's supported override.

```bash
UV_NO_BINARY_PACKAGE=tokenizers .venv-tools/bin/python -m src.lock_dependencies
```

The current tests cover fresh lock directory creation, platform and browser
constraints, selection of the tooling environment's resolver, and explicit
failure reporting. After an explicit runtime setup, check the installed
environment and audit each platform's resolved packages.

```bash
.venv/bin/python -m pip check
.venv/bin/python -m src.environment
.venv-tools/bin/python -m pip_audit -r locks/py311-macos-arm64.txt -r locks/py311-macos-arm64-browser.txt --disable-pip
.venv-tools/bin/python -m pip_audit -r locks/py311-linux-x86_64.txt -r locks/py311-linux-x86_64-browser.txt --disable-pip
```

### Remaining dependency and runtime risks

The November 5, 2026 audit reports no advisories for the selected PyTorch and
Requests versions. This addresses CVE-2025-32434 and CVE-2024-47081. Both platform
audits still report 50 advisory entries across six other packages.

| Package | Pinned version | Advisory entries |
|---|---|---|
| datasets | 2.21.0 | 1 |
| fsspec | 2024.6.1 | 1 |
| pyarrow | 17.0.0 | 2 |
| scikit-learn | 1.2.2 | 2 |
| sentence-transformers | 3.0.1 | 1 |
| transformers | 4.46.3 | 43 |

These remaining packages were not upgraded as part of the two dependency
repairs. Advisory matches do not establish an exploitable path through this
pipeline, but the environment must not be described as free of known
vulnerabilities. Recheck current advisories before release.

The isolated macOS installation passed hash checked installation, dependency
consistency, runtime lock verification, imports, and pinned model CPU embedding
inference. Linux locks were resolved and audited, and compatible nonyanked
wheels with matching hashes were confirmed for all 96 packages. Linux execution
was not verified because no running Linux container runtime was available.

A clustering smoke check also exposed PEP 3118 buffer errors inside
`pdc-dp-means` and scikit-learn. The same errors reproduce in the unchanged
original environment. This preexisting issue remains unresolved, so successful
embedding inference does not establish that the full pipeline works correctly.

## Local material and sharing

The local `final-replication/` directory is excluded through `.gitignore`.
It may contain request logs, saved responses, copied source, results, provenance,
and machine specific paths. It is not part of the maintained project.

Ignore rules do not delete files or sanitize a manually copied directory.
Move that experiment directory elsewhere before sharing the whole workspace.
Also exclude local environments, credentials, caches, and generated outputs.

### User-supplied inputs

Review your inputs for sensitive information and confirm that you have the
necessary permissions before processing or sharing them. Configured hosted
models receive dataset text during inference. Keep private inputs and generated
outputs out of shared copies of the workspace.

Code licensing is handled separately by the maintainer. Include the separately
prepared licenses and required notices before public release.

### Source code

* [`src/disagreement_analysis/`](src/disagreement_analysis/) — the
  per-stage implementations (`preprocessor_disagreement.py`,
  `disagreement_explainer.py`, `evaluate.py`, …).
* [`src/dashboard/disagreement_dashboard.py`](src/dashboard/disagreement_dashboard.py)
  — the three-pane settings UI rendered by the CLI.

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos
in modified versions of this project must not cause confusion or imply Microsoft
sponsorship. Any use of third-party trademarks or logos are subject to those
third-party's policies.
