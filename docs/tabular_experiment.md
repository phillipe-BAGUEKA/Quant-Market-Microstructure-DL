# Qualification et première expérience tabulaire

La tâche reste l'identification de 24 actions anonymisées à partir de 100
événements. Une classification correcte ne démontre ni rentabilité, ni détection
d'investisseurs, ni représentation universelle des marchés.

## Contrôles et référence

`baselines/reference.py` constitue un oracle de calcul sur de petites observations
déjà acceptées par le validateur de production. Il n'appelle aucun réducteur de
production : `math.fsum`, `statistics.pstdev` et `Counter` produisent les valeurs
attendues. Ce n'est pas un deuxième validateur exhaustif des entrées.

Les tolérances sont fixées avant comparaison : `rtol=1e-6`, `atol=1e-6`, pour
accepter quelques unités d'arrondi en float32, avec une borne absolue près de
zéro. Les noms, l'ordre, les catégories, les identifiants, labels et indices
sources sont contrôlés exactement. Un exemple affine non monotone distingue
les six colonnes et les six statistiques, notamment `ddof=0` et first/last.

L'échantillon réel comprend trois observations de développement par classe :
première, médiane et dernière dans l'ordre source de y. Les événements complets
sont capturés pendant le scan unique. Chaque observation est comparée avec les
vocabulaires des cinq trains. C'est une preuve numérique sur cet échantillon,
pas une équivalence exhaustive de chaque valeur de chaque observation.

## Exécution reproductible

Depuis la racine, avec un nouveau nom de répertoire pour chaque tentative :

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -u -m cfm_challenge.baselines.experiment --output reports/experiments/tabular-01 --cache data/processed/tabular-summaries.npz --chunk-size 20000 --pilot-observations 2000 --max-iter 2000 --threads 2
```

B0 utilise une matrice constante de signature, pas des features de marché.
Le pilote lit au plus 2000 observations complètes, avec anticipation du lecteur
par chunks (au plus 220000 lignes parsées pour chunk_size=20000). Il vérifie
durée et mémoire entre observations ; la limite de 120 secondes est coopérative,
pas une interruption d'un appel natif en cours.

La préparation principale effectue un seul parcours complet de X, traversant
physiquement les lignes du lockbox mais ne les résumant pas. La correspondance
X/y, l'unicité, les séquences de 100 événements et l'ordre sont contrôlés. La
progression est affichée et journalisée, au moins toutes les 1000 observations
ou environ 15 secondes, hors appel bloquant. Ce coût de journalisation fait
partie du temps mesuré. Limites coopératives : 3600 secondes de préparation,
RSS de 2 Gio, mémoire disponible minimale de 384 Mio. La composition est
contrôlée à son retour ; l'entraînement natif n'est pas interrompu par ce garde.

Les cinq matrices sont composées et évaluées séquentiellement. Chaque vocabulaire
est appris sur son train ; chaque scaler est ajusté par le pipeline uniquement
sur ce même train. Un problème de convergence est archivé puis arrête les folds
suivants. Une nouvelle tentative peut augmenter uniquement le plafond numérique
`--max-iter`, sans changer les partitions ni la régularisation.

## Cache et preuves

Le cache dans `data/processed/`, ignoré par Git, utilise NPZ sans pickle :
statistiques float64, fréquences, comptes de venues, identifiants, labels et
indices sources. Les vecteurs de référence échantillonnés sont stockés séparément
dans un fichier `.oracle.npz`, également ignoré. SHA-256 vérifie l'intégrité des
deux fichiers ; les structures et la sélection déterministe sont revalidées.
Un cache incomplet n'est jamais présenté comme prêt pour l'évaluation ; seul
son checkpoint peut être repris pour terminer la préparation et les contrôles.

Après un arrêt mémoire de la première tentative, le stockage de préparation a
été durci : les résumés sont maintenant écrits dans des tableaux `.npy` mappés
sur disque, dans le répertoire `.packed`. Cela évite de conserver des centaines
de milliers de tuples et flottants Python. Les valeurs sont flushées avant les
marqueurs de lignes terminées, notamment avant les contrôles de ressources.
Après un arrêt contrôlé, une reprise de ce checkpoint relit X pour vérifier la
structure et capturer l'échantillon, mais ne recalcule pas les résumés terminés.
Ce mécanisme ne prétend pas assurer une transaction atomique lors d'une panne
matérielle. Après préparation complète, les tableaux sont également protégés
par SHA-256 et servent à la composition des folds avec une mémoire limitée.
Le NPZ final reste un format d'échange indépendant de ces fichiers de travail.

La provenance enregistre les empreintes du code actuel, y compris les fichiers
non suivis, de y et des partitions. L'identité de X utilise taille et mtime :
c'est une garde pratique, **pas une empreinte cryptographique du contenu de X**.
Ne pas réutiliser le cache après remplacement de X, même à métadonnées égales.
Une empreinte complète de X n'est pas calculée par un scan supplémentaire.

Les rapports JSON et JSONL contiennent les métriques par fold, supports, matrices
de confusion, configuration, versions, état Git, temps et mémoire. Les moyennes
et écarts types (`ddof=0`) sont descriptifs. Aucun score officiel d'agrégation ni
macro-F1 de prédictions hors fold regroupées n'est revendiqué.

## Suite du programme

Le programme comprend B0, régression logistique, CNN 1D, GRU bidirectionnel,
CNN–LSTM (direction à concevoir), TCN, InceptionTime et MiniRocket multivarié
avec un classifieur compatible avec les métriques. CatBoost est exclu. TCN,
InceptionTime et MiniRocket sont étudiés séparément, en parallèle du développement ;
cet apprentissage ne constitue plus un prérequis à leur implémentation.

Les résumés tabulaires ne reconstruisent pas les événements ordonnés. Le
[jalon séquentiel](sequences.md) conserve les 100 événements, canaux numériques,
catégories brutes, identifiants et indices sources dans des tableaux mappés.
Il est qualifié sur des données synthétiques et son cache complet de 128 640
observations a été construit et contrôlé le 4 novembre 2026. Les vocabulaires
et normalisations restent
propres au train. Ce cache nécessite une nouvelle lecture de X et ne matérialise
ni embeddings globaux ni cinq copies des données.

Le rôle C++ proposé est la préparation commune par lots, choisie après profilage
séparant lecture, assemblage, conversions et transformations par événement.
Mesurer séparément le gain du cache, du calcul C++ et de l'interface ; aucun de
ces gains n'est automatiquement un gain d'entraînement. Aucun compilateur,
binding ou nouveau modèle n'est installé ou implémenté pendant ce jalon.

### Contrat séquentiel : implémentation et qualification bornée

- Une ligne logique par observation de développement, avec correspondance exacte
  entre identifiant, label et indice source ; axe événement de longueur 100,
  dans l'ordre réellement lu, sans tri déduit de `obs_id`.
- Six canaux numériques bruts conservés initialement en float64 pour éviter une
  nouvelle perte de précision ; catégories séparées et typées, sans les traiter
  comme des grandeurs continues. Toute conversion ultérieure en float32 devra
  être qualifiée avant l'entraînement.
- Les codes bruts de venue sont conservés sans vocabulaire global ajusté. Les
  encodages, catégories inconnues et normalisations sont appris sur chaque train.
- Tableaux `.npy` mappables et métadonnées de provenance, lus par petits lots ;
  aucun embedding ni cinq copies complètes des séquences ne sont persistés.

Pour 128640 observations, les six canaux float64 représentent environ 589 Mio.
Une venue int64 et trois catégories uint8 validées représentent environ 135 Mio
supplémentaires. Ce sont des tailles de stockage des valeurs, pas une estimation
de la RAM d'entraînement : activations, lots, gradients et états d'optimisation
s'ajoutent. Les codes et bornes exacts seront vérifiés avant tout choix de dtype.

La qualification synthétique et les pilotes CPU bornés CNN 1D/GRU bidirectionnel
sont terminés ; voir le [bilan séquentiel](../reports/experiments/20261103-sequence-results.md).
Le cache complet a ensuite été construit en 385,70 secondes, avec contrôles
mémoire et d'alignement ; le [nouveau bilan](../reports/experiments/20261104-sequence-results.md)
est séparé des pilotes historiques.
La [reprise du fold 0](../reports/experiments/20261104-sequence-retry-results.md)
a ensuite terminé une époque CNN et BiGRU, avec validation complète. B1 reste
la référence tabulaire convergée sur cinq folds ; les réseaux ne sont qualifiés
qu'à une époque sur un seul fold, sans conclusion définitive sur les architectures.
Le coût mesuré des agrégats tabulaires ne suffit pas à désigner le goulot du
pipeline séquentiel ; le profilage de celui-ci est rapporté séparément.

Trois questions de compréhension pour relire ce jalon :

1. Pourquoi conserver les comptes bruts des venues de validation n'est-il pas
   une fuite tant qu'ils ne choisissent pas le vocabulaire du train ?
2. Que prouve la comparaison indépendante des 72 observations, et que ne
   permet-elle pas d'affirmer sur les 128640 observations ?
3. Pourquoi les 50 agrégats et le futur axe de 100 événements ne sont-ils pas
   deux représentations temporelles interchangeables ?
