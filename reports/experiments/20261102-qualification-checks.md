# Vérifications du jalon tabulaire — 2 novembre 2026

État Git initial : branche main, HEAD et origin/main local
`1e1fc35fabeb84713ba714b283ab5617c76f911f`. Modifications préexistantes :
pyproject.toml et les sept fichiers baselines/tests non suivis. Aucun fetch,
git add, commit, push ou changement de branche.

Toutes les commandes Python utilisent `.\.venv\Scripts\python.exe` avec
`PYTHONDONTWRITEBYTECODE=1`. Les tests utilisent des fichiers synthétiques
temporaires, avec accès autorisé hors bac à sable pour pytest.

| Commande | Code | Résultat |
|---|---:|---|
| `python -m pytest -p no:cacheprovider tests/unit/test_baseline_qualification.py -q --tb=short` (première version) | 0 | 5 passed in 5.55s |
| `python -m pytest -p no:cacheprovider tests/unit/test_baseline_qualification.py tests/unit/test_baseline_aggregation.py tests/unit/test_baseline_dataset.py tests/unit/test_tabular_baseline.py -q --tb=short` | 0 | 179 passed in 5.22s |
| `python -m pytest -p no:cacheprovider tests/unit -q --tb=short` (avant revue) | 0 | 265 passed in 6.28s |
| `python -m pytest -p no:cacheprovider tests/unit/test_baseline_qualification.py -q --tb=short` (après corrections) | 0 | 14 passed in 4.07s |
| `python -m pytest -p no:cacheprovider tests/unit -q --tb=short` (après corrections) | 0 | 273 passed in 7.90s |
| `python -m cfm_challenge.baselines.experiment --help` | 0 | CLI importée, options affichées |
| `git diff --check` | 0 | Aucun défaut d'espacement suivi ; avertissement LF/CRLF pour pyproject.toml |

Les nouvelles sources non suivies ont également été inspectées séparément :
aucun espace final relevé. Le contrôle Git ne couvre pas leur contenu.

## Unique revue indépendante

Agent personnalisé reviewer, lecture seule, sans sous-agent ni CSV réel.

Corrections retenues : validation des formes, types, finitude et indices du
cache oracle ; contrôle de durée utilisant le vrai départ de composition ;
déclaration directe de threadpoolctl déjà installé ; confinement des chemins
d'écriture aux répertoires reports et data/processed ; présentation des lignes
les plus longues. Les tests ciblés et complets ont été relancés après correction.

Limites conservées explicitement : X est identifié par taille et mtime, pas par
SHA-256 de son contenu ; l'oracle indépendant vérifie les calculs sur des entrées
déjà validées et ne remplace pas le validateur ; l'échantillon réel ne prouve pas
l'équivalence numérique exhaustive. Le test de non-appel des réducteurs est une
garde limitée, complétée par la lecture du module sans import d'aggregation et
par les résultats analytiques calculés séparément.

## Première commande réelle

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:GIT_OPTIONAL_LOCKS='0'
.\.venv\Scripts\python.exe -u -m cfm_challenge.baselines.experiment --output reports/experiments/20261102-tabular-200 --cache data/processed/20261102-tabular-summaries.npz --chunk-size 100000 --pilot-observations 2000 --max-prepare-seconds 3600 --max-iter 200 --threads 2
```

Les sorties réelles et tout arrêt sont conservés dans les fichiers JSON/JSONL du
répertoire de cette tentative. Ce document de tests ne préjuge pas de sa réussite.

## Arrêt mémoire et correction bornée

La première commande réelle a quitté avec le code 1 : moins de 384 Mio de RAM
disponible. Dernier jalon de progression archivé : 136000 observations physiques,
108822 résumés de développement ; arrêt à 818,63 secondes depuis le début de
l'exécuteur. Le nombre exact traité entre ce jalon et l'arrêt n'était pas loggé.
Pic du processus : 530,82 Mio. B0 est archivée, aucun B1 n'a été lancé. Le cache
complet n'existait pas ; les résumés partiels de cette première version ont été
perdus. Aucune équivalence complète ni préparation réussie n'est revendiquée.

La correction remplace le stockage Python croissant par des tableaux numériques
mappés sur disque, tout en conservant les mêmes calculs. Un test démontre une
reprise après arrêt contrôlé : le lecteur revalide les lignes et l'agrégateur
n'est appelé que pour les observations dont le résumé manque. Les tests du
stockage, des catégories inconnues, des frontières de chunks et de l'oracle
indépendant continuent de passer.

- Tests ciblés après correction mémoire : `15 passed in 19.43s`, code 0.
- Suite complète après correction mémoire : `274 passed in 28.35s`, code 0.
- Aucun deuxième reviewer lancé : la revue indépendante reste unique ; cette
  correction consécutive au blocage réel a été relue et testée par l'agent principal.

Nouvelle commande réelle, avec chunks réduits et le stockage durci :

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:GIT_OPTIONAL_LOCKS='0'
.\.venv\Scripts\python.exe -u -m cfm_challenge.baselines.experiment --output reports/experiments/20261102-tabular-packed-200 --cache data/processed/20261102-tabular-summaries.npz --chunk-size 20000 --pilot-observations 2000 --max-prepare-seconds 3600 --max-iter 200 --threads 2
```

Cette tentative implique un nouveau parcours après le préfixe interrompu ; il
ne faut pas présenter l'ensemble des tentatives comme une unique lecture de X.

## Limite de durée et reprise du checkpoint

La deuxième tentative s'est arrêtée avec `RuntimeError: resource stop:
preparation time budget exceeded`. Le journal confirme 153775 observations
physiques contrôlées et 123062 résumés persistés, à 3663,11 secondes depuis le
démarrage. Le budget de 3600 secondes concerne la préparation après le pilote.
Le pic résident du processus est de 284,11 Mio. Le code numérique de sortie de
cette session n'a pas été récupéré après expiration de la session d'exécution ;
l'échec et son motif sont archivés, aucune réussite n'est inférée.

Reprise sans changement du code ni de la provenance du checkpoint :

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:GIT_OPTIONAL_LOCKS='0'
.\.venv\Scripts\python.exe -u -m cfm_challenge.baselines.experiment --output reports/experiments/20261102-tabular-resume-200 --cache data/processed/20261102-tabular-summaries.npz --chunk-size 20000 --pilot-observations 2000 --max-prepare-seconds 3600 --max-iter 200 --threads 2
```

La reprise revalide physiquement le CSV et recapture l'échantillon indépendant,
mais ne recalcule que les 5578 résumés manquants. Elle n'est donc pas un deuxième
calcul complet des agrégats et ne répète pas le scan pour chaque fold. Le coût
réel total inclut néanmoins les deux préfixes interrompus, cette relecture et
les petits pilotes ; l'objectif d'une seule lecture physique sur l'ensemble
des tentatives n'a pas été atteint.

Comparaison SHA-256 des 37 fichiers préexistants relevés au départ : seuls
`pyproject.toml` et `src/cfm_challenge/baselines/dataset.py` ont changé. Les
autres sources, anciens tests, README, LICENSE, règles Git et rapports
préexistants ont été préservés.

## Qualification réelle terminée et plafond de convergence

La reprise a achevé la préparation : 160800 observations physiques, 16080000
événements, 128640 résumés de développement. Durée de cette préparation de
reprise et de sa qualification : 646,56 secondes. Les 72 observations complètes
déterministes ont été comparées avec cinq vocabulaires, soit 360 vecteurs de
50 features. Écart absolu maximal : 2,7755575615628914e-17 ; tolérances fixées
avant lecture : rtol=1e-6 et atol=1e-6. Le cache final existe et son SHA-256
est `5c4b6ba8f575ecc6aec0e7f7e9f1b0ca2ec9e14c853a6ddbc5f4dff9c2846dae`.

La commande de reprise a ensuite quitté avec le code 1 : B1 fold 0 atteint
200 itérations sans convergence. Cette tentative ne valide pas B1. Son résultat
et l'avertissement complet sont conservés dans `20261102-tabular-resume-200` :
accuracy 0,278023943 ; macro-F1 0,266803629 ; log-loss 2,288364302.
Composition 44,76 s, fit 65,83 s. Aucun des quatre folds suivants n'a alors été
entraîné. Pic résident global de cette tentative : 647,23 Mio.

Correction numérique : seul le plafond d'itérations passe de 200 à 2000.
Le solveur, la tolérance, la régularisation, les features et les partitions
restent identiques. Le cache qualifié est réutilisé sans lecture de X :

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:GIT_OPTIONAL_LOCKS='0'
.\.venv\Scripts\python.exe -u -m cfm_challenge.baselines.experiment --output reports/experiments/20261102-tabular-final-2000 --cache data/processed/20261102-tabular-summaries.npz --chunk-size 20000 --pilot-observations 2000 --max-prepare-seconds 3600 --max-iter 2000 --threads 2
```

Le résultat de cette dernière commande est détaillé dans le rapport final du
jalon, avec ses métriques par fold ; ce plafond n'impose pas 2000 itérations
si le critère de convergence est atteint auparavant.

## Vérifications finales terminées

- Commande `20261102-tabular-final-2000` : code 0, 830,23 s, cinq folds
  convergés en 351, 353, 319, 269 et 346 itérations, aucun avertissement.
- B1 descriptif : accuracy 0,278016169 ; macro-F1 0,266697627 ; log-loss
  2,282292131. Résultats complets dans `20261102-tabular-results.md`.
- Relecture automatisée des cinq JSON : supports de 1072 par classe, matrices
  24 × 24 dont les sommes de lignes correspondent aux supports, valeurs finies,
  effectifs 102912/25728, convergence ; code 0.
- Empreintes des sources et tests comparées au manifeste réel : identiques ;
  SHA-256 du cache comparé au fichier d'intégrité : identique ; code 0.
- Imports des modules de qualification, cache et exécution : code 0.
- `git diff --check` final : code 0 ; seul avertissement LF/CRLF habituel.
- Huit fichiers de sources/tests/documentation/rapports vérifiés séparément :
  aucun espace final ni tabulation d'indentation ; code 0.
- Recherche ciblée de chemins personnels, adresses électroniques et motifs de
  secrets dans les nouveaux fichiers et rapports : aucun résultat. Cela ne
  remplace pas une revue de sécurité exhaustive des travaux préexistants.
- Branche, HEAD et origin/main local inchangés. Working tree volontairement
  non propre : changements préexistants conservés et nouveaux fichiers du jalon.
- Aucun git add, commit, push, fetch, pull, merge, rebase ni changement de branche.
