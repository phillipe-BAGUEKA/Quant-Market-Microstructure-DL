# CFM Challenge — signatures de microstructure

Identifier **une action anonymisée parmi 24 classes** à partir de **100 événements
de marché**. Le projet étudie l'information portée par les cotations, quantités,
venues, types d'événements et leur ordre.

Une classification correcte ne démontre ni utilité de trading, ni détection
d'investisseurs, ni représentation universelle. Ces applications potentielles
restent des hypothèses distinctes à tester.

## État vérifié

| Composant | État |
|---|---|
| Schémas, lecture par morceaux, audit X/y | Implémentés et testés |
| Lockbox, folds stratifiés, remappage des indices | Implémentés et utilisés |
| 50 features, résumés réutilisables, cache avec reprise | Implémentés ; contrôles structurels réels et qualification numérique échantillonnée |
| B0 : DummyClassifier(strategy="prior") | Évaluée sur cinq folds avec vrais labels |
| B1 : StandardScaler + régression logistique | Évaluée et convergée sur cinq folds réels |
| Folds par blocs candidats | Implémentés pour sensibilité ; aucune campagne B0/B1 complète avec ce découpage |
| Cache de séquences ordonnées | Complet et contrôlé : 128 640 observations de développement, 100 événements chacune |
| CNN 1D, GRU bidirectionnel | Pilotes CPU disponibles ; première époque CNN interrompue par le garde mémoire, aucune validation séquentielle complète |
| CNN–LSTM, TCN, InceptionTime, MiniRocket multivarié | Prévus ; pas encore implémentés |
| Composante C++ | Prévue après profilage ; aucun noyau implémenté |

CatBoost est exclu. L'étude théorique de TCN, InceptionTime et MiniRocket avance
séparément, en parallèle du développement ; elle ne le bloque pas.
La direction du CNN–LSTM sera précisée lors de sa conception.

## Données et limites

Un événement est une ligne de X. Une observation regroupe 100 événements et
porte un label. `obs_id` relie X à `eqt_code_cat` dans y.

| Ensemble | Effectif |
|---|---:|
| X_train | 16 080 000 événements, 160 800 observations |
| Classes | 24, de 0 à 23 ; 6 700 observations chacune |
| Développement | 128 640 observations |
| Lockbox réservé | 32 160 observations |

Placer les fichiers du challenge dans `data/raw/` : ils ne sont pas fournis par
ce dépôt. Données et caches sont ignorés par Git. Leur licence est distincte
de la licence MIT du code ; aucun transfert cloud n'est autorisé par défaut.

L'ordre source des événements est conservé. `obs_id` ne prouve pas une
chronologie globale. Le test officiel, décrit comme futur, n'est pas utilisé
dans ces expériences. Voir le [contrat sémantique](docs/challenge_data_contract.md)
et le [profil historique agrégé](reports/train_feature_profile.md).

## Protocole et métriques

`SplitConfiguration()` réserve un lockbox stratifié de 20 %, puis crée cinq
folds stratifiés sur le développement, graine 42. Chaque fold entraîne un
nouveau modèle sur **102 912 observations** et le valide sur **25 728**.
Les validations sont disjointes et couvrent une fois le développement.

Vocabulaire des venues et scaler sont appris uniquement sur le train du fold.
Le lockbox reste réservé, sans entraînement ni évaluation. Les 50 features
sont des résumés d'une observation, **pas 50 instants d'une séquence**.

La **cross-entropy multiclasse** et la **log-loss** correspondent à la moyenne
de `-log(p(classe correcte))`, avec logarithme naturel, labels exclusifs, sans
pondération et même réduction. PyTorch `CrossEntropyLoss` reçoit des logits
et applique log-softmax ; scikit-learn reçoit des probabilités normalisées.
Clipping numérique, pondération et réduction peuvent modifier l'égalité.

Nous suivons aussi accuracy, macro-F1, supports et matrices de confusion.
Ces métriques internes ne définissent pas un score officiel confirmé.
Voir le [protocole détaillé](docs/validation_protocol.md).

## Résultats B0/B1 vérifiés

Expérience du 2 novembre 2026 ; les cinq folds B1 ont convergé sans avertissement.

| Modèle / fold | Accuracy | Macro-F1 | Log-loss |
|---|---:|---:|---:|
| B0 — chaque fold | 0,041667 | 0,003333 | 3,178054 |
| B1 — 0 | 0,279345 | 0,268133 | 2,286558 |
| B1 — 1 | 0,273321 | 0,262597 | 2,280209 |
| B1 — 2 | 0,280006 | 0,268194 | 2,282545 |
| B1 — 3 | 0,280317 | 0,268537 | 2,268772 |
| B1 — 4 | 0,277091 | 0,266027 | 2,293377 |
| B1 — moyenne descriptive | **0,278016** | **0,266698** | **2,282292** |

La moyenne des macro-F1 ne coïncide pas nécessairement avec la macro-F1 calculée
sur les prédictions hors fold réunies.
Les dispersions ne sont pas des intervalles de confiance de cinq expériences
indépendantes. Le [rapport complet](reports/experiments/20261102-tabular-results.md)
décrit les incidents, temps, mémoire et limites. Le
[manifeste original](reports/experiments/20261102-tabular-final-2000/manifest.json)
conserve les paramètres et empreintes réellement exécutés.
Les modèles ajustés n'ont pas été sérialisés ; les résultats sont conservés.

Les [pilotes séquentiels historiques](reports/experiments/20261103-sequence-results.md)
vérifient les dimensions, la loss et la rétropropagation sur un petit train.
Le [jalon suivant](reports/experiments/20261104-sequence-results.md) a construit
et contrôlé le cache complet en 385,70 s. La première tentative d'époque CNN
s'est arrêtée à 82 688 observations de train quand la RAM disponible est passée
sous 384 Mio. Il n'existe pas encore de score séquentiel de validation comparable
à B1 ; une loss partielle de train ne le remplace pas.

## Installation et contrôles courts

Environnement vérifié : Windows, Python 3.12 64 bits. Les métadonnées autorisent
Python >=3.12 ; les autres versions ne sont pas toutes qualifiées.
Ne pas écraser un environnement existant.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit -q
.\.venv\Scripts\python.exe -m cfm_challenge.baselines.experiment --help
```

Dépendances tabulaires : NumPy, pandas, scikit-learn, threadpoolctl ; pytest est
dans l'extra `dev`. psutil est facultatif : sans lui les mesures mémoire sont
signalées indisponibles. L'activation PowerShell n'est pas nécessaire.
Les tests sont synthétiques et n'ouvrent pas les CSV réels.

Le [rapport d'environnement](reports/environment.md) et le
[freeze historique](reports/environment-freeze.txt) décrivent l'installation
initiale, dont PyTorch CPU. Un freeze ne conserve pas l'index CPU :
installer torch depuis `https://download.pytorch.org/whl/cpu` avant de reproduire
ce relevé. Les versions des expériences sont celles de leurs manifestes.

Les modules séquentiels demandent l'extra optionnel `sequences` (torch, psutil).
Installer d'abord torch depuis l'index CPU si souhaité, puis `pip install -e
".[dev,sequences]"` depuis PyPI ; voir les commandes exactes du
[guide séquentiel](docs/sequences.md). Les tests des réseaux sont ignorés si torch
n'est pas installé ; cela ne constitue pas une validation de ces réseaux.

## Reproduire B0/B1

Cette commande est **coûteuse** si aucun cache compatible n'existe. Choisir un
nouveau dossier de sortie et, après changement de code, un nouveau cache.
Ne pas la lancer comme simple vérification de documentation.

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -u -m cfm_challenge.baselines.experiment --output reports/experiments/tabular-new-run --cache data/processed/tabular-new-run.npz --chunk-size 20000 --pilot-observations 2000 --max-prepare-seconds 3600 --max-iter 2000 --threads 2
```

Le pilote est borné. La préparation conserve des résumés indépendants des folds,
puis compose et évalue les matrices séquentiellement. Une reprise peut relire X
après un arrêt, sans recalculer les résumés terminés. Les anciens chemins à deux
scans par fold restent disponibles, mais ne sont pas recommandés pour cette campagne.

La provenance stricte invalide le cache après modification du code, même par
formatage. Les expériences historiques ne sont pas réattribuées au code
reformaté : voir la [note de publication](docs/publication.md) et le
[guide d'expérience](docs/tabular_experiment.md).

## Organisation

```text
src/cfm_challenge/
  config.py                 chemins du projet
  data/                     schémas, lecture, audit, profilage agrégé
  evaluation/               partitions et contrôles des folds
  baselines/
    aggregation.py          résumés et 50 features
    dataset.py              sélection et datasets par fold
    tabular.py              factories, fit et métriques B0/B1
    reference.py            oracle numérique indépendant
    summary_cache.py        stockage et reprise
    experiment.py           campagne observable et archivage
  sequences/
    cache.py                séquences brutes ordonnées et reprise
    models.py               encodeur train-only, petits CNN/GRU pilotes
    pilot.py                pilote CPU borné, sans score de validation
    experiment.py           une époque CPU sur le fold 0 depuis le cache complet
tests/unit/                 tests synthétiques
docs/                       contrats et guides actuels
reports/                    preuves historiques
data/raw/, data/processed/  données et caches locaux ignorés
```

Les modules `data.audit`, `data.profiling` et `evaluation.split_audit` exposent
`--help`. Leurs audits réels peuvent être coûteux. Ne pas écraser leurs rapports
historiques pour actualiser l'état courant.

## Prochaines étapes

1. Libérer de la RAM avant la reprise de l'expérience : viser 2,5 Gio disponibles.
   Le cache complet est terminé et réutilisable, sans nouveau scan X.
2. Terminer une époque CNN 1D/GRU sur le fold 0 avec le budget de dix minutes
   par modèle, puis comparer les métriques de validation à B1 sur ce même fold.
3. Comparer les autres familles retenues avec des budgets explicités.
4. Profiler lecture/assemblage, transformations, chargement et réseau avant
   de choisir un noyau C++ commun aux modèles séquentiels.

Les résumés tabulaires ne permettent pas de reconstruire les séquences :
un nouveau parcours est nécessaire. Aucun gain C++, entraînement complet
supplémentaire ou dépense cloud n'est présumé.
