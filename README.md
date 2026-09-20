# CFM Market Microstructure Challenge

This repository is the foundation for a modular Python, PyTorch, and modern C++
solution to the CFM market microstructure classification challenge. The current
implementation is deliberately limited to project configuration and explicit
schema contracts. It contains no model or training code.

## Data contract

An **event** is one row of the feature table. An **observation** is identified
by `obs_id` and contains an ordered sequence of events. A **class** is the
integer target `eqt_code_cat` associated with one observation.

The following facts were verified locally from the provided CSV files:

- `X_train.csv` contains 16,080,000 events and 160,800 observations.
- Every training observation contains exactly 100 events.
- `y_train.csv` contains 24 classes, numbered from 0 through 23.
- Every class contains 6,700 training observations.
- `X_test.csv` contains 8,160,000 events and 81,600 observations.
- `obs_id` is the key relating feature and target files.
- `eqt_code_cat` is the multiclass target.

The numeric order of `obs_id` is not assumed to represent chronology. Event
order must be preserved from the source until the challenge documentation and
validation protocol establish any stronger rule.

Raw challenge data lives under `data/raw/` and is excluded from Git. The MIT
license in this repository applies to the code; rights and conditions for the
challenge data remain separate and must be checked before redistribution or
transfer to third-party infrastructure.

## Planned architecture

The Python package currently provides schema validation and streaming training
data ingestion. Future work may add encoding, PyTorch datasets, reproducible
training, evaluation, inference, and submission validation. Planned model
families include simple baselines, ANN, CNN 1D, LSTM, and Transformer
architectures; none is implemented yet.

A future C++ component may handle a genuinely stateful sequential operation if
profiling shows that it is useful. It will require an up-to-date toolchain,
equivalence tests against a Python reference, and a benchmark before any speed
claim. Python integration may later use pybind11 if direct calls are justified.

The project uses scripts and importable modules rather than Jupyter notebooks.
Development and small experiments should run locally first. A CPU or GPU cloud
environment will only be considered after measuring runtime, memory, and data
transfer needs and confirming the applicable challenge rules.

## Local installation

Python 3.12 or newer is required. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Pandas is the only runtime dependency of the current package. The optional
`dev` extra provides pytest for the unit tests. Future PyTorch dependencies
will be declared when package code actually uses them.
For a CPU-only PyTorch environment, the PyTorch CPU index must be recorded
explicitly because a package freeze alone does not preserve the source index.

## Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit -q
```

The unit tests use only synthetic Python values and temporary paths. They do
not read the raw CSV files.

## Streaming ingestion and audit

`cfm_challenge.data.loading` validates the complete X and y headers before
reading data. It loads y into a validated `obs_id` to label mapping, then reads
X through pandas chunks. It never sorts by `obs_id`: source-row order remains
the event order of every emitted observation.

The implementation relies on, and checks, contiguity of events belonging to
one observation. When an observation crosses a chunk boundary, only its
incomplete trailing rows are retained and combined with the next chunk. A
sequence must close with exactly 100 events. The reader rejects missing or
duplicate labels, missing observation identifiers, lengths other than 100,
non-contiguous reappearance of an identifier, X observations without labels,
and labels unused by X. The final X/y coverage check occurs when the consumer
fully exhausts the labeled-observation generator; the audit command does so.

Run an aggregate-only audit without writing output files:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m cfm_challenge.data.audit --chunk-size 100000
```

The command reports event and observation counts, class counts, and elapsed
time. It does not print source rows, identifiers, prices, or label-to-identifier
associations. It does not establish a maximum memory measurement; its memory
guarantee is limited to the current chunk, the label map, closed identifiers,
and one incomplete observation.

## Current scope and open points

Implemented now:

- an installable `src/` package;
- an immutable, path-based project configuration;
- constants and validation functions for the observed X and y schemas;
- chunked X ingestion with contiguous-observation assembly;
- validated label loading and X/y coverage checks;
- an aggregate-only command-line audit;
- focused unit tests for configuration and schema validation.

Still to confirm before model selection or final validation are the official
ranking metric, the submission constraints, dependencies between observations,
and any chronology available beyond the row order inside each observation.
Compact data types also require explicit range and precision checks. No score,
normalization, feature engineering, train/validation split, tensor, trained
model, training loop, C++ component, or submission is present.

## Data semantics and aggregate profiling

The official column contract and benchmark reference are paraphrased in
`docs/challenge_data_contract.md`; the benchmark is documented only. Regenerate
an aggregate-only profile without retaining the complete feature CSV in memory:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m cfm_challenge.data.profiling --chunk-size 100000 --output reports/train_feature_profile.md
```

The report separates official rules from local aggregate observations. It never
prints source rows or individual identifiers, and it does not correct data.
