# Protocole de validation interne

Selon la [page officielle ChallengeData 146](https://challengedata.ens.fr/challenges/146), le test provient d'une période future. Les classes de train sont équilibrées, mais les fichiers locaux ne fournissent pas de date explicite. Les identifiants sont ordonnés localement; les blocs candidats de 480 ne sont pas équilibrés à 20 observations par classe.

Le développement réserve un lockbox stratifié de 20 %, puis applique cinq folds stratifiés au développement. Un second test de sensibilité utilise des `candidate_block_id` contigus, sans leur donner une signification calendaire.

Le lockbox mesure une généralisation i.i.d., pas temporelle. Les blocs candidats ne sont pas des journées démontrées et aucun protocole interne ne reproduit parfaitement la période future du test.

Les métriques internes futures seront accuracy, macro-F1, CrossEntropy/log-loss, matrice de confusion et accuracy par classe. CrossEntropy est la loss du benchmark, pas une métrique officielle confirmée. Une classe constante vaut 1/24 ≈ 4,17 %; une distribution uniforme a une CrossEntropy de log(24) ≈ 3,178.
