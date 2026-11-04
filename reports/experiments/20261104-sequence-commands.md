# Commandes du jalon séquentiel — 4 novembre 2026

## Préparation complète

Commande réellement exécutée, à lancer depuis la racine du dépôt. Pour une nouvelle exécution,
choisir un nouveau dossier de rapport. Le dossier de cache peut être réutilisé uniquement
si ses métadonnées correspondent ; une reprise conserve les observations terminées.
Les durées internes commencent après les imports Python.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import psutil

from cfm_challenge.baselines.dataset import build_development_selection, localize_folds
from cfm_challenge.config import ProjectConfig
from cfm_challenge.data.loading import load_labels
from cfm_challenge.evaluation.splitting import SplitConfiguration, build_lockbox, stratified_folds
from cfm_challenge.sequences.cache import build_sequence_cache

output = Path("reports/experiments/20261104-sequence-cache")
directory = Path("data/processed/20261104-sequences-development")
output.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
process = psutil.Process()

def emit(stage, **values):
    memory = process.memory_info()
    record = {
        "stage": stage,
        "total_seconds": time.perf_counter() - started,
        "rss_mib": memory.rss / 2**20,
        "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 2**20,
        "available_mib": psutil.virtual_memory().available / 2**20,
        **values,
    }
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    if stage != "cache_progress" or values.get("processed", 0) % 5000 == 0 or values.get("selected_ready") == 128640:
        print(json.dumps(record, allow_nan=False), flush=True)
    if record["available_mib"] < 384:
        raise RuntimeError("Available RAM below 384 MiB; controlled stop")
    return record

paths = ProjectConfig(Path.cwd())
configuration = SplitConfiguration()
y_path = paths.raw_data_dir / "y_train.csv"
x_path = paths.raw_data_dir / "X_train.csv"
emit("start")
labels_by_id = load_labels(y_path)
all_ids = tuple(labels_by_id)
all_labels = tuple(labels_by_id.values())
partition = build_lockbox(all_ids, all_labels, configuration)
selection = build_development_selection(y_path, lockbox_fraction=configuration.lockbox_fraction, seed=configuration.seed)
global_folds = stratified_folds(all_labels, partition, configuration)
folds = localize_folds(global_folds, selection.source_indices)
lockbox_ids = {all_ids[index] for index in partition.lockbox_indices}
assert len(selection.observation_ids) == 128640
assert len(partition.lockbox_indices) == 32160
assert not set(selection.observation_ids) & lockbox_ids
np.testing.assert_array_equal(selection.source_indices, partition.development_indices)
assert all(len(f.train_indices) == 102912 and len(f.validation_indices) == 25728 for f in folds)
assert len(folds) == 5
source_sha = hashlib.sha256(selection.source_indices.astype("<i8").tobytes()).hexdigest()
y_sha = hashlib.sha256(y_path.read_bytes()).hexdigest()
historical = json.loads(Path("reports/experiments/20261102-tabular-final-2000/manifest.json").read_text())
manifest = {
    "split": asdict(configuration),
    "rows": len(selection.observation_ids),
    "lockbox_rows": len(partition.lockbox_indices),
    "chunk_size": 20000,
    "max_seconds": 600,
    "initial_minimum_available_mib": 1024,
    "running_minimum_available_mib": 384,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "psutil": psutil.__version__,
    "x_size": x_path.stat().st_size,
    "x_mtime_ns": x_path.stat().st_mtime_ns,
    "y_sha256": y_sha,
    "source_indices_sha256": source_sha,
    "cache_code_sha256": hashlib.sha256(Path("src/cfm_challenge/sequences/cache.py").read_bytes()).hexdigest(),
    "cache_directory": directory.as_posix(),
    "scope": "development only; no learned encoder, training, validation score or lockbox evaluation",
}
assert y_sha == "0fcb04d1b6b7c94de6b10db594ac16d1ef60a43bef8a52e4f20214699dd514a2"
assert source_sha == "2b2c6ca3c9a6c12d5d4636729951c153ea9420ce4026cf90aa10a22025ad4a11"
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
emit("y_only_validated", development_rows=len(selection.labels), lockbox_rows=len(partition.lockbox_indices), folds=len(folds), train_rows=102912, validation_rows=25728, source_sha_matches_B1=True)
if psutil.virtual_memory().available < 1024 * 2**20:
    raise RuntimeError("Less than 1 GiB available before cache preparation")
if psutil.disk_usage(".").free < 2 * 2**30:
    raise RuntimeError("Less than 2 GiB free on disk")
prepare_start = time.perf_counter()
emit("cache_start")
try:
    cache = build_sequence_cache(
        x_path, selection, directory, chunk_size=20000, max_seconds=600,
        observer=lambda values: emit("cache_progress", **values),
    )
    preparation_seconds = time.perf_counter() - prepare_start
    np.testing.assert_array_equal(cache.observation_ids, selection.observation_ids)
    np.testing.assert_array_equal(cache.labels, selection.labels)
    np.testing.assert_array_equal(cache.source_indices, selection.source_indices)
    assert not set(cache.observation_ids.tolist()) & lockbox_ids
    assert cache.numeric.shape == (128640, 100, 6)
    assert cache.venues.shape == (128640, 100)
    assert cache.categorical.shape == (128640, 100, 3)
    assert cache.numeric.dtype == np.float64 and cache.venues.dtype == np.int64
    assert cache.categorical.dtype == np.uint8
    for start in range(0, len(cache.labels), 64):
        numeric = cache.numeric[start:start + 64]
        categorical = cache.categorical[start:start + 64]
        assert np.isfinite(numeric).all()
        for column, count in enumerate((3, 2, 2)):
            assert (categorical[:, :, column] < count).all()
    ready = np.load(directory / "ready.npy", mmap_mode="r", allow_pickle=False)
    assert ready.all()
    hashes = {}
    for path in sorted(directory.glob("*")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(2**20):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    result = emit("complete", preparation_seconds=preparation_seconds, total_rows=128640,
        event_count=128640 * 100, alignment_checked=True, finite_numeric_checked=True,
        categorical_bounds_checked=True, lockbox_excluded=True, event_order="source order retained by iter_observations and direct ordered copy; synthetic exact-order regression test",
        cache_bytes=sum(p.stat().st_size for p in directory.glob("*")), cache_sha256=hashes,
        models_trained=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
except Exception as error:
    ready_path = directory / "ready.npy"
    completed = int(np.load(ready_path, mmap_mode="r", allow_pickle=False).sum()) if ready_path.exists() else 0
    print(json.dumps({"stage": "stopped", "selected_ready": completed, "exception": type(error).__name__, "error": str(error)}), flush=True)
    raise

'@ | .\.venv\Scripts\python.exe -
```

## Époque complète du fold 0

Commande réellement exécutée après réussite des tests synthétiques et de la revue.
Le chronomètre PowerShell inclut le démarrage Python ; les chronomètres internes
séparent les étapes après imports.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$sequenceTimer = [System.Diagnostics.Stopwatch]::StartNew()
.\.venv\Scripts\python.exe -u -m cfm_challenge.sequences.experiment --cache data/processed/20261104-sequences-development --output reports/experiments/20261104-sequence-fold0-epoch
$sequenceExitCode = $LASTEXITCODE
$sequenceTimer.Stop()
Write-Output "EXPERIMENT_EXIT_CODE=$sequenceExitCode WALL_SECONDS=$($sequenceTimer.Elapsed.TotalSeconds)"
exit $sequenceExitCode
```

## Codes de sortie et contrôles

| Commande / contrôle | Code | Résultat |
|---|---:|---|
| Préparation ci-dessus | 0 | 128 640 observations mises en cache ; 385,695 s de préparation |
| Expérience ci-dessus | 1 | Arrêt mémoire à 82 688 lignes train CNN ; aucune validation |
| Quatre nouveaux tests | 0 | 4 passed in 10.60s |
| Trois fichiers de tests séquentiels | 0 | 14 passed in 8.60s |
| Suite unitaire | 0 | 288 passed in 16.92s |
| CLI `--help` | 0 | Import et arguments disponibles |
| Black `--check`, deux nouveaux fichiers | 0 | Deux fichiers inchangés |
| `git diff --check` | 0 | Aucune erreur d'espacement ; avertissements LF/CRLF seulement |

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit/test_sequence_experiment.py -q
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit/test_sequence_cache.py tests/unit/test_sequence_models.py tests/unit/test_sequence_experiment.py -q
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit -q
.\.venv\Scripts\python.exe -m cfm_challenge.sequences.experiment --help
git diff --check
git status --short --untracked-files=all
git check-ignore -v data/raw/X_train.csv data/raw/X_test.csv data/raw/y_train.csv data/raw/y_random.csv data/processed/20261104-sequences-development/numeric.npy .venv/Scripts/python.exe
git ls-remote --heads origin main
```

Les tests ont utilisé l'accès autorisé aux répertoires temporaires après un
refus d'accès du bac à sable. Black 25.1.0 était déjà dans le dossier d'outils
temporaire isolé : aucune installation de dépendance n'a eu lieu.

Un contrôle Python séparé a lu les treize fichiers modifiés ou nouveaux :
espaces de fin de ligne, tabulations, fin de fichier, analyse AST des Python,
parsing JSON/JSONL, motifs de secrets/chemins personnels et conformité des
empreintes du code au manifeste exécuté. Il a aussi vérifié trente liens locaux.
Deux fins de fichiers Markdown ont été corrigées (ligne blanche finale), puis
ce contrôle a réussi avec code 0 et aucune anomalie. Ce contrôle est nécessaire
car `git diff --check` ne couvre pas encore les fichiers non suivis.
