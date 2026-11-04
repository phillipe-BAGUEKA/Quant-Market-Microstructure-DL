# Commande de reprise du pilote fold 0 — 4 novembre 2026

## Résultats des commandes

| Commande | Code de sortie | Résultat |
|---|---:|---|
| Première reprise | 1 | CNN terminé ; BiGRU arrêté pendant le train après une veille Windows ; 693,978 s |
| Deuxième précontrôle | 1 | RAM insuffisante après imports ; aucun entraînement ; 8,405 s |
| Troisième reprise | 0 | CNN et BiGRU terminés ; 470,982 s, dont 170,023 s d'attente mémoire |
| Tests séquentiels existants | 0 | 14 passed in 18.45s |
| Contrôles JSON/JSONL, liens et espacement | 0 | 21 fichiers et 49 liens locaux contrôlés |
| `git diff --check` | 0 | Aucun défaut d'espacement ; avertissements LF/CRLF seulement |

Commande de tests :

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit/test_sequence_cache.py tests/unit/test_sequence_models.py tests/unit/test_sequence_experiment.py -q
```

Les accès nécessaires aux dossiers temporaires de ces tests ont été autorisés.
Le code n'a pas changé ; la suite unitaire complète n'a pas été relancée.

## Première reprise

Commande réellement exécutée depuis la racine du dépôt. Utiliser un autre dossier
pour une nouvelle tentative. Le seuil initial est vérifié après les imports et
le contrôle des empreintes du cache. Le garde de 384 Mio et le budget de 600 s
par modèle restent ceux du lanceur existant. Aucun poids n'est restauré.
Un audit Python interdit toute ouverture de X_train, X_test ou y_random ;
le contrôle de provenance de X utilise seulement ses métadonnées.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$retryTimer = [System.Diagnostics.Stopwatch]::StartNew()
@'
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

from cfm_challenge.sequences.experiment import run

output = Path("reports/experiments/20261104-sequence-fold0-retry")
cache = Path("data/processed/20261104-sequences-development")
if output.exists():
    raise RuntimeError("A new output directory is required")

previous = json.loads(Path("reports/experiments/20261104-sequence-fold0-epoch/manifest.json").read_text())
for name, expected in previous["code_sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Code differs from the qualified previous attempt: " + name)

qualified = json.loads(Path("reports/experiments/20261104-sequence-cache/result.json").read_text())
hash_start = time.perf_counter()
for name, expected in qualified["cache_sha256"].items():
    digest = hashlib.sha256()
    with (cache / name).open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError("Qualified cache differs: " + name)
hash_seconds = time.perf_counter() - hash_start

def forbid_raw_x(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        name = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).name.lower()
        if name in {"x_train.csv", "x_test.csv", "y_random.csv"}:
            raise RuntimeError("Raw X/test reads forbidden in this cached retry")

sys.addaudithook(forbid_raw_x)
preflight = {
    "available_mib": psutil.virtual_memory().available / 2**20,
    "required_initial_mib": 2560,
    "running_guard_mib": 384,
    "cache_files_sha256_match": True,
    "cache_hash_seconds": hash_seconds,
    "code_unchanged_since_first_attempt": True,
    "raw_x_and_test_open_guard": True,
    "from_initialization": True,
    "cache_directory": cache.as_posix(),
}
print(json.dumps({"stage": "preflight", **preflight}), flush=True)
if preflight["available_mib"] < preflight["required_initial_mib"]:
    Path("reports/experiments/20261104-sequence-retry-preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    raise RuntimeError("Less than 2.5 GiB available; model run not started")
started = time.perf_counter()
code = 1
try:
    run(output, cache)
    code = 0
finally:
    if output.exists():
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        execution = {"exit_code": code, "run_wall_seconds_after_preflight": time.perf_counter() - started}
        (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "execution_exit", **execution}), flush=True)

'@ | .\.venv\Scripts\python.exe -
$retryExitCode = $LASTEXITCODE
$retryTimer.Stop()
Write-Output "EXPERIMENT_EXIT_CODE=$retryExitCode WALL_SECONDS=$($retryTimer.Elapsed.TotalSeconds)"
exit $retryExitCode
```

## Relance bloquée avant entraînement

Après la veille due au capot, une seconde commande a contrôlé le même cache.
Elle s'est arrêtée avant `run` : **2 223,89 Mio disponibles**, seuil 2 560 Mio,
code 1, durée PowerShell 8,405 s. Aucun modèle n'a été recréé dans cette commande.
Son [précontrôle](20261104-sequence-retry-2-preflight.json) est conservé.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$retryTimer = [System.Diagnostics.Stopwatch]::StartNew()
@'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

from cfm_challenge.sequences.experiment import run

output = Path("reports/experiments/20261104-sequence-fold0-retry-2")
cache = Path("data/processed/20261104-sequences-development")
if output.exists():
    raise RuntimeError("A new output directory is required")

previous = json.loads(Path("reports/experiments/20261104-sequence-fold0-epoch/manifest.json").read_text())
for name, expected in previous["code_sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Code differs from the qualified previous attempt: " + name)

qualified = json.loads(Path("reports/experiments/20261104-sequence-cache/result.json").read_text())
hash_start = time.perf_counter()
for name, expected in qualified["cache_sha256"].items():
    digest = hashlib.sha256()
    with (cache / name).open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError("Qualified cache differs: " + name)
hash_seconds = time.perf_counter() - hash_start

def forbid_raw_x(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        name = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).name.lower()
        if name in {"x_train.csv", "x_test.csv", "y_random.csv"}:
            raise RuntimeError("Raw X/test reads forbidden in this cached retry")

sys.addaudithook(forbid_raw_x)
preflight = {
    "utc_start": datetime.now(timezone.utc).isoformat(),
    "available_mib": psutil.virtual_memory().available / 2**20,
    "required_initial_mib": 2560,
    "running_guard_mib": 384,
    "cache_files_sha256_match": True,
    "cache_hash_seconds": hash_seconds,
    "code_unchanged_since_first_attempt": True,
    "raw_x_and_test_open_guard": True,
    "from_initialization": True,
    "cache_directory": cache.as_posix(),
}
print(json.dumps({"stage": "preflight", **preflight}), flush=True)
if preflight["available_mib"] < preflight["required_initial_mib"]:
    Path("reports/experiments/20261104-sequence-retry-2-preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    raise RuntimeError("Less than 2.5 GiB available; model run not started")
started = time.perf_counter()
code = 1
try:
    run(output, cache)
    code = 0
finally:
    if output.exists():
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        execution = {"utc_end": datetime.now(timezone.utc).isoformat(), "exit_code": code, "run_wall_seconds_after_preflight": time.perf_counter() - started}
        (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "execution_exit", **execution}), flush=True)

'@ | .\.venv\Scripts\python.exe -
$retryExitCode = $LASTEXITCODE
$retryTimer.Stop()
Write-Output "EXPERIMENT_EXIT_CODE=$retryExitCode WALL_SECONDS=$($retryTimer.Elapsed.TotalSeconds)"
exit $retryExitCode
```

## Attente mémoire avant une troisième tentative

Cette variante conserve les mêmes données et paramètres. Elle vérifie la RAM
après imports pendant au plus trois minutes, et demande deux mesures successives
au-dessus de 2 560 Mio avant d'appeler le lanceur. Aucune attente ne modifie le
plafond de dix minutes par modèle.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$retryTimer = [System.Diagnostics.Stopwatch]::StartNew()
@'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

from cfm_challenge.sequences.experiment import run

output = Path("reports/experiments/20261104-sequence-fold0-retry-3")
cache = Path("data/processed/20261104-sequences-development")
if output.exists():
    raise RuntimeError("A new output directory is required")

previous = json.loads(Path("reports/experiments/20261104-sequence-fold0-epoch/manifest.json").read_text())
for name, expected in previous["code_sha256"].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Code differs from the qualified previous attempt: " + name)

qualified = json.loads(Path("reports/experiments/20261104-sequence-cache/result.json").read_text())
hash_start = time.perf_counter()
for name, expected in qualified["cache_sha256"].items():
    digest = hashlib.sha256()
    with (cache / name).open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError("Qualified cache differs: " + name)
hash_seconds = time.perf_counter() - hash_start

def forbid_raw_x(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        name = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).name.lower()
        if name in {"x_train.csv", "x_test.csv", "y_random.csv"}:
            raise RuntimeError("Raw X/test reads forbidden in this cached retry")

sys.addaudithook(forbid_raw_x)
preflight = {
    "utc_start": datetime.now(timezone.utc).isoformat(),
    "available_mib": psutil.virtual_memory().available / 2**20,
    "required_initial_mib": 2560,
    "running_guard_mib": 384,
    "cache_files_sha256_match": True,
    "cache_hash_seconds": hash_seconds,
    "code_unchanged_since_first_attempt": True,
    "raw_x_and_test_open_guard": True,
    "from_initialization": True,
    "cache_directory": cache.as_posix(),
}
memory_wait_started = time.perf_counter()
memory_wait_samples = []
consecutive_ready = 0
while True:
    available = psutil.virtual_memory().available / 2**20
    memory_wait_samples.append({"seconds": time.perf_counter() - memory_wait_started, "available_mib": available})
    consecutive_ready = consecutive_ready + 1 if available >= 2560 else 0
    print(json.dumps({"stage": "memory_wait", **memory_wait_samples[-1]}), flush=True)
    if consecutive_ready == 2 or time.perf_counter() - memory_wait_started >= 180:
        break
    time.sleep(10)
preflight["available_mib"] = available
preflight["utc_start"] = datetime.now(timezone.utc).isoformat()
preflight["memory_wait_samples"] = memory_wait_samples
preflight["memory_wait_seconds"] = time.perf_counter() - memory_wait_started
print(json.dumps({"stage": "preflight", **preflight}), flush=True)
if preflight["available_mib"] < preflight["required_initial_mib"]:
    Path("reports/experiments/20261104-sequence-retry-3-preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    raise RuntimeError("Less than 2.5 GiB available; model run not started")
started = time.perf_counter()
code = 1
try:
    run(output, cache)
    code = 0
finally:
    if output.exists():
        (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        execution = {"utc_end": datetime.now(timezone.utc).isoformat(), "exit_code": code, "run_wall_seconds_after_preflight": time.perf_counter() - started}
        (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "execution_exit", **execution}), flush=True)

'@ | .\.venv\Scripts\python.exe -
$retryExitCode = $LASTEXITCODE
$retryTimer.Stop()
Write-Output "EXPERIMENT_EXIT_CODE=$retryExitCode WALL_SECONDS=$($retryTimer.Elapsed.TotalSeconds)"
exit $retryExitCode
```
