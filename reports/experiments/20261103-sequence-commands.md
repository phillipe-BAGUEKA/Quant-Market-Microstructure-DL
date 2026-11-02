# Commandes des pilotes séquentiels — 3 novembre 2026

Les dossiers préfixés 20261102 conservent le nom choisi au début de la session.
Cette note est rédigée le 3 novembre, heure Europe/Kiev. Les rapports B0/B1
historiques restent inchangés.

## Tests et pilote terminés

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit/test_sequence_cache.py tests/unit/test_sequence_models.py -q --tb=short
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit -q --tb=short
.\.venv\Scripts\python.exe -u -m cfm_challenge.sequences.pilot --output reports/experiments/20261102-sequence-pilot --cache data/processed/20261102-sequence-pilot
```

Codes de sortie 0 : 10 passed in 9.24s ; 284 passed in 8.95s ; pilote complet.
Les tests utilisent l'accès autorisé aux répertoires temporaires. PyTorch CPU et
psutil étaient déjà installés ; aucun paquet runtime n'a été installé ici.

## Préparation complète : arrêt avant X

Une première tentative s'est arrêtée avant X car la RAM disponible était sous
1 Gio (code 1 ; valeur exacte non journalisée, voir preflight.json).
Une mesure ultérieure hors processus était remontée à 1195,05 Mio. La tentative
suivante a conservé le même seuil ; après chargement des modules et de y, elle
a mesuré **952,71875 Mio disponibles**, puis s'est arrêtée avant X, code 1.
Aucun cache complet ni checkpoint partiel n'a été créé.

Commande exacte de cette seconde tentative ; elle est coûteuse si le garde
mémoire autorise son démarrage. Pour la rejouer, choisir un nouveau répertoire
de rapport. Ne pas l'utiliser comme vérification de documentation.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
@'
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
from cfm_challenge.baselines.dataset import build_development_selection
from cfm_challenge.evaluation.splitting import SplitConfiguration
from cfm_challenge.sequences.cache import build_sequence_cache
output = Path('reports/experiments/20261102-sequence-cache-retry')
output.mkdir(exist_ok=False)
start = time.perf_counter()
selection = build_development_selection('data/raw/y_train.csv')
def emit(values):
    memory = psutil.Process().memory_info()
    record = {'elapsed_seconds_total': time.perf_counter()-start, 'rss_mib': memory.rss/2**20, 'peak_rss_mib': memory.peak_wset/2**20, 'available_mib':psutil.virtual_memory().available/2**20, **values}
    with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
    if values.get('processed', 0) % 1000 == 0 or values.get('selected_ready') == 128640:
        print(json.dumps(record), flush=True)
    if record['available_mib'] < 384:
        raise RuntimeError('Available memory below 384 MiB')
emit({'stage':'preflight'})
if psutil.virtual_memory().available < 1024 * 2**20:
    raise RuntimeError('Less than 1 GiB available before full cache preparation')
assert len(selection.observation_ids) == 128640
config = {'split': asdict(SplitConfiguration()), 'chunk_size':20000, 'max_seconds':600, 'rows':128640, 'scope':'development raw sequences; no fitted transformation', 'cache_code_sha256':hashlib.sha256(Path('src/cfm_challenge/sequences/cache.py').read_bytes()).hexdigest()}
(output/'manifest.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
try:
    cache = build_sequence_cache('data/raw/X_train.csv', selection, 'data/processed/20261102-sequences-development', chunk_size=20000, max_seconds=600, observer=emit)
    np.testing.assert_array_equal(cache.observation_ids, selection.observation_ids)
    np.testing.assert_array_equal(cache.labels, selection.labels)
    np.testing.assert_array_equal(cache.source_indices, selection.source_indices)
    result = {'stage':'complete','rows':len(cache.labels),'numeric_shape':list(cache.numeric.shape),'venues_shape':list(cache.venues.shape),'categorical_shape':list(cache.categorical.shape),'total_seconds':time.perf_counter()-start,'cache_bytes':sum(p.stat().st_size for p in Path('data/processed/20261102-sequences-development').glob('*')),'lockbox_excluded':True,'models_trained':False}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    emit(result)
except Exception as error:
    emit({'stage':'stopped','error':str(error)})
    raise
'@ | .\.venv\Scripts\python.exe -
```
