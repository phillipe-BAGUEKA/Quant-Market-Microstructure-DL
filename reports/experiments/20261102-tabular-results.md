# Qualification réelle et baselines B0/B1 — 2 novembre 2026

## Résultat terminé

Les cinq folds B1 sont composés, contrôlés et évalués ; ils ont tous convergé,
sans avertissement de convergence. La dernière commande quitte avec le code 0.
B0 est également reproduite sur les cinq partitions avec les vrais labels.
Aucune évaluation du lockbox ni du jeu de test officiel n'a été effectuée.

Les métriques détaillées, supports et matrices de confusion sont dans
[20261102-tabular-final-2000](20261102-tabular-final-2000/).
Le [manifeste](20261102-tabular-final-2000/manifest.json) conserve les versions,
paramètres, empreintes des partitions et de tous les fichiers Python exécutés,
y compris les sources non commitées. Le [journal](20261102-tabular-final-2000/progress.jsonl)
conserve les étapes et ressources observées.

## Protocole effectif

- 160800 observations étiquetées, 24 classes, 6700 observations par classe.
- Lockbox stratifié : fraction 0,2, graine 42, 32160 observations réservées.
- Développement : 128640 observations ; cinq folds stratifiés, graine 42.
- Chaque fold : 102912 lignes train et 25728 validation ; support de validation
  exactement 1072 pour chacune des 24 classes.
- Ordre des classes : les entiers de 0 à 23.
- Matrice propre à chaque fold : 128640 × 50, float32, valeurs finies,
  ordre exact des identifiants, labels et indices sources contrôlé.
- Les validations sont disjointes et couvrent exactement le développement.
  Les indices globaux sont localisés explicitement ; le lockbox est exclu.
- Vocabulaire calculé uniquement sur les résumés du train de chaque fold :
  six venues connues dans chacun des cinq cas réels. Le scaler est ajusté par
  le pipeline exclusivement sur les lignes train, comme dans les tests.
- Aucun identifiant individuel ni vecteur réel n'est reproduit dans ce rapport.

B1 conserve la factory existante : StandardScaler puis LogisticRegression,
solveur lbfgs, C=1, tol=0,0001, class_weight=None, random_state=42. Seul le
plafond max_iter a été porté de 200 à 2000 après une non-convergence archivée.
Aucun changement de régularisation, de features, de labels ou de partitions.
Deux threads natifs sont autorisés par threadpoolctl, déjà installé.

## Résultats par fold

B0 : DummyClassifier(strategy="prior"), matrice constante de signature
128640 × 1, qui n'est pas une extraction des événements de marché.
Sur chacun des folds 0, 1, 2, 3 et 4 :
accuracy **0,041666667**, macro-F1 **0,003333333**, log-loss **3,178053830**.
Ces valeurs reproduisent le contrôle historique du fold réel 0.

| Fold B1 | Accuracy | Macro-F1 | Log-loss | Itérations | Composition (s) | Fit (s) | Prédiction + métriques (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.279345460 | 0.268133267 | 2.286557989 | 351 | 43.75 | 112.21 | 0.484 |
| 1 | 0.273320896 | 0.262596816 | 2.280208969 | 353 | 44.08 | 108.39 | 0.498 |
| 2 | 0.280006219 | 0.268194009 | 2.282544569 | 319 | 43.22 | 99.43 | 0.474 |
| 3 | 0.280317164 | 0.268537235 | 2.268772160 | 269 | 43.27 | 84.51 | 0.501 |
| 4 | 0.277091107 | 0.266026809 | 2.293376970 | 346 | 45.54 | 108.27 | 0.517 |

Les cinq matrices de confusion ont la forme 24 × 24. Les effectifs train,
validation et supports par classe sont identiques aux effectifs annoncés
ci-dessus ; leurs valeurs exactes et les matrices complètes sont archivées
dans les fichiers b0_fold_*.json et b1_fold_*.json.

Résumé **descriptif** de B1, moyenne ± écart type de population (ddof=0) :

- accuracy : 0.278016169 ± 0.002604735 ;
- macro-F1 : 0.266697627 ± 0.002233877 ;
- log-loss : 2.282292131 ± 0.008102643.

Il ne s'agit pas d'une agrégation officielle du challenge. La moyenne des
macro-F1 n'est pas nécessairement la macro-F1 des prédictions hors fold réunies.
Les folds se partagent des lignes d'entraînement : ces écarts types ne sont
pas des intervalles de confiance fondés sur cinq expériences indépendantes.

## Qualification des features

Le nouvel oracle est indépendant des réducteurs de production : math.fsum,
statistics.pstdev et Counter. Les exemples synthétiques distinguent les six
colonnes numériques, les statistiques, first/last et ddof=0. Les noms et leur
ordre, dtype, catégories inconnues, vocabulaire de train, frontières de chunks,
alignement et stockage/reprise sont testés.

Tolérances arrêtées avant comparaison : rtol=1e-6, atol=1e-6. Elles permettent
quelques unités d'arrondi float32 et une petite tolérance absolue près de zéro.
Les identifiants, noms, classes, comptes et indices sont contrôlés exactement.

Pendant le parcours réel, 72 observations complètes ont été conservées en
mémoire : première, médiane et dernière observation de développement de chaque
classe, dans l'ordre source de y. Cinq vocabulaires donnent 360 vecteurs comparés,
soit 18000 valeurs. Écart absolu maximal observé :
**2,7755575615628914e-17**.

Cette comparaison est **échantillonnée**, non une équivalence numérique
exhaustive des 128640 observations. En revanche, le lecteur et les contrôles
du parcours complet vérifient la correspondance X/y, l'unicité des observations,
les 100 événements et l'ordre ; chaque matrice finale est contrôlée pour sa
forme, sa finitude et son alignement.

Les lignes lockbox sont physiquement traversées par le lecteur de X, mais ne
sont pas résumées ni utilisées pour ajuster les transformations ou noter B1.
La validation reste stratifiée et non chronologique ; obs_id n'est pas une
preuve de chronologie globale ou de généralisation à une autre période.

## Temps, mémoire et incidents conservés

| Tentative | Résultat / code | Mesures et dernier jalon |
|---|---|---|
| tabular-200 | Échec, code 1 | Garde RAM disponible <384 Mio ; arrêt à 818,63 s ; dernier jalon 136000 observations et 108822 résumés ; pic 530,82 Mio ; pas de cache récupérable |
| tabular-packed-200 | Échec RuntimeError, code numérique non récupéré après expiration de session | Budget de préparation dépassé ; arrêt à 3663,11 s ; checkpoint exact : 153775 observations parcourues, 123062 résumés persistés ; pic 284,11 Mio |
| tabular-resume-200 | Préparation qualifiée, puis échec convergence B1, code 1 | 5578 résumés manquants calculés ; préparation/relecture/qualification 646,56 s ; arrêt global à 833,06 s ; pic 647,23 Mio |
| tabular-final-2000 | Réussite, code 0 | Cinq folds convergés, aucun scan X ; 830.23 s ; pic 656.73 Mio |

Le fold 0 non convergé à 200 itérations reste archivé comme tentative distincte :
accuracy 0,278023943, macro-F1 0,266803629, log-loss 2,288364302 ; il n'est pas
mélangé aux cinq résultats convergés.

Les pilotes bornés de 2000 observations ont duré respectivement 18,94 s,
38,59 s et 41,11 s ; le calcul des résumés y occupait 15,31 s, 31,55 s et
33,89 s. Ces mesures locales ne sont pas des prévisions garanties du scan
complet et ne prouvent pas la cause historique de l'écart entre 18 et 96 min.

La reprise réussie a physiquement lu les 160800 observations et réutilisé
123062 résumés. **Le total des tentatives n'est pas un scan unique** : deux
préfixes interrompus, un parcours achevé, et trois petits pilotes. Les données
ont été relues après les arrêts, jamais pour chacun des cinq folds B1.

Le cache final est partagé : NPZ de 60099234 octets, oracle de 79916 octets,
tableaux .npy mappés et petit manifeste. Les tableaux évitent une accumulation
de tuples Python en RAM. La reprise garantit les arrêts contrôlés, pas une
transaction atomique lors d'une panne matérielle.

Pour la tentative finale : 219.86 s cumulées de composition,
512.80 s cumulées de fit. La mémoire disponible minimale **aux jalons
journalisés** était 2307.40 Mio ; ce n'est pas un minimum
continu garanti. Le pic résident est celui rapporté par Windows/psutil.

## Commandes, tests et traçabilité

Les commandes exactes de chaque tentative et leurs incidents figurent dans
[le journal des vérifications](20261102-qualification-checks.md).

Dernière exécution réelle réussie :

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:GIT_OPTIONAL_LOCKS='0'
.\.venv\Scripts\python.exe -u -m cfm_challenge.baselines.experiment --output reports/experiments/20261102-tabular-final-2000 --cache data/processed/20261102-tabular-summaries.npz --chunk-size 20000 --pilot-observations 2000 --max-prepare-seconds 3600 --max-iter 2000 --threads 2
```

Code 0. Cette commande exige un dossier de résultats neuf. Le cache de même
provenance permet de reproduire les évaluations sans relire X ; changer un
fichier Python suivi par le manifeste invalide volontairement cette provenance.

Derniers tests après la correction mémoire :

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit/test_baseline_qualification.py -q --tb=short
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit -q --tb=short
```

- 15 passed in 19.43s, code 0.
- 274 passed in 28.35s, code 0.
- Imports reference/cache/experiment : code 0.
- git diff --check : code 0, avertissement LF/CRLF sur pyproject.toml seulement.
- Contrôle séparé des sources/tests/docs non suivis : aucun espace final ni
  tabulation d'indentation trouvé ; le diff Git seul ne couvre pas ces fichiers.
- git check-ignore confirme l'exclusion du NPZ, de l'oracle et des tableaux mappés.

Le manifeste et les fichiers d'intégrité utilisent SHA-256 pour le code, y,
les partitions et les caches. **X est identifié seulement par taille et mtime**,
sans scan de hachage supplémentaire : ce n'est pas une garantie cryptographique
du contenu source. Un remplacement de X à métadonnées identiques invaliderait
l'hypothèse de réutilisation.

Une extraction administrative du tableau récapitulatif a d'abord quitté avec
le code 1 (clé JSON confusion_matrix inexistante), puis avec le code 0 après
emploi du champ réel confusion. Aucun code de calcul, résultat ou test n'a été
modifié par cette correction de lecture du rapport.

## Modifications et état Git

État initial et final de branche : main, HEAD et origin/main local
1e1fc35fabeb84713ba714b283ab5617c76f911f. Aucun fetch ou changement de référence.

Ordre logique des changements :

1. reference.py : oracle numérique indépendant.
2. summary_cache.py : cache sans pickle et checkpoint numérique mappé.
3. dataset.py : observateur des observations complètes et stockage optionnel,
   sans changer les réducteurs existants ni les partitions.
4. experiment.py : qualification, reprise, provenance, contrôles et évaluation
   séquentielle des baselines existantes.
5. test_baseline_qualification.py : quinze cas de qualification et de reprise.
6. pyproject.toml : déclaration directe de threadpoolctl déjà installé ; la
   modification numpy préexistante a été conservée. Aucune installation.
7. docs/tabular_experiment.md et reports/experiments : protocole, preuves,
   résultats et proposition du prochain cache.

Comparaison des empreintes initiales : seuls pyproject.toml et dataset.py
parmi les 37 fichiers préexistants inspectés ont changé. Les autres changements
sont de nouveaux fichiers listés ci-dessus. Les sept fichiers de tranche 5
déjà non suivis restent non suivis ; aucune modification préexistante supprimée.
Les artefacts numériques sont sous data/processed, ignoré par Git.

Une seule revue indépendante a été effectuée. Les corrections retenues
concernent validation du cache oracle, minuteur, chemins de sortie et dépendance
directe. Les limites de provenance X et de portée de l'oracle restent explicites.
La correction mémoire postérieure à cette revue a été relue et testée par
l'agent principal, sans second reviewer. La revue complète des changements
accumulés en vue d'un commit reste une étape distincte.

## Prochain jalon recommandé

Le contrat et les tailles du futur cache sont proposés dans
[la documentation](../../docs/tabular_experiment.md). Il conservera les
100 événements ordonnés, canaux numériques et catégories brutes séparées,
identifiants/labels/indices sources ; vocabulaire et normalisation train-only.
Les 50 agrégats ne sont pas 50 instants et ne permettent pas de reconstruire
les séquences. Un nouveau parcours sera nécessaire pour ce cache.

À titre de budget de stockage pour le développement : environ 589 Mio pour
six canaux float64, plus 135 Mio pour venue int64 et trois catégories uint8
après validation. Utiliser des tableaux mappés, des petits lots et aucun
embedding global persisté. La mémoire d'entraînement s'ajoute à ces volumes.

Qualifier ce cache puis effectuer un pilote CNN 1D et GRU bidirectionnel avant
les expériences suivantes. Le programme TCN, InceptionTime et MiniRocket reste
à comprendre puis réaliser séparément ; CatBoost reste exclu.

Le C++ doit viser une préparation commune effectivement coûteuse après profilage
(assemblage, conversions, calculs par événement), sans réécrire d'emblée le lecteur.
Le coût tabulaire observé ne désigne pas le futur goulot séquentiel. Mesurer
séparément le gain du cache, du noyau compilé et de l'interface, sans confondre
préparation et entraînement.

Seule décision immédiate proposée : le budget matériel/temps des futurs pilotes.
Recommandation : qualifier le cache et mesurer un petit pilote sur CPU local,
puis décider d'un éventuel GPU avec une estimation mesurée ; aucune dépense ni
installation n'est engagée ici.

B0_FIVE_FOLDS_LABEL_ONLY_VALIDATED=true
B1_FIVE_FOLDS_EVALUATED=true
B1_ALL_FOLDS_CONVERGED=true
REAL_NUMERICAL_QUALIFICATION=deterministic_sample_only
READY_FOR_COMMIT=false
