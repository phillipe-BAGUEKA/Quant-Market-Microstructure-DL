# Protocole de validation interne

Selon la [page officielle ChallengeData 146](https://challengedata.ens.fr/challenges/146), le test provient d'une période future. Les classes de train sont équilibrées, mais les fichiers locaux ne fournissent pas de date explicite. Les identifiants sont ordonnés localement; les blocs candidats de 480 ne sont pas équilibrés à 20 observations par classe.

Le développement réserve un lockbox stratifié de 20 %, puis applique cinq folds stratifiés au développement. Un second test de sensibilité utilise des `candidate_block_id` contigus, sans leur donner une signification calendaire.

Le lockbox est réservé à une évaluation finale interne ; il n'a pas été évalué
pendant B0/B1. Son découpage aléatoire ne démontre pas l'indépendance des
observations ni une généralisation temporelle. Les blocs candidats ne sont pas
des journées démontrées et aucun protocole interne ne reproduit parfaitement
la période future du test.

## Protocole exécuté pour B0/B1

| Paramètre | Valeur |
|---|---:|
| Graine | 42 |
| Fraction lockbox | 0,2 |
| Nombre de folds stratifiés | 5 |
| Développement / lockbox | 128640 / 32160 |
| Train / validation par fold | 102912 / 25728 |
| Classes dans l'ordre | 0 à 23 |

`build_lockbox` et `stratified_folds` produisent des indices globaux. La fonction
`localize_folds` les convertit vers les lignes compactées du développement.
Les validations couvrent ces lignes exactement une fois, sans chevauchement.
Les venues et le scaler sont appris uniquement sur le train du fold courant.

Le chemin recommandé compose les 50 features depuis les résumés indépendants
des folds. `iter_fold_datasets` reste le chemin à deux scans de X par fold ;
`iter_fold_datasets_from_summaries` évite ces scans répétés. Le constructeur
`build_development_dataset`, qui apprend un vocabulaire sur tout le développement,
ne doit pas servir à une cross-validation stricte de ce vocabulaire.

Les métriques effectivement archivées sont accuracy, macro-F1, log-loss, matrice
de confusion, accuracy et support par classe. Pour des labels exclusifs, sans
pondération, la cross-entropy moyenne et la log-loss mesurent `-log(p_classe)`.
CrossEntropyLoss attend des logits, log_loss des probabilités ; les conventions
numériques peuvent différer. La cross-entropy du benchmark ne confirme pas à
elle seule une métrique officielle de classement.

Une classe constante donne 1/24 ≈ 4,17 % sur ces validations équilibrées ; une
distribution uniforme donne log(24) ≈ 3,178. Les moyennes sur folds sont
descriptives, pas une agrégation officielle ou des prédictions hors fold réunies.
Voir les [résultats vérifiés](../reports/experiments/20261102-tabular-results.md).
