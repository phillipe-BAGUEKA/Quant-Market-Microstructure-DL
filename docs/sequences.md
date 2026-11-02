# Cache ordonné et pilotes CPU

Ce jalon est séparé de B0/B1, publié dans le commit
`30f3534cfd6e2d640e99d29d660488593146fc7d`. Il ne reproduit pas exactement
l'architecture du benchmark officiel et ne constitue pas une campagne à cinq folds.

## Contrat du cache

`sequences.cache.build_sequence_cache` reçoit une `DevelopmentSelection` explicite.
Il ne décide ni des partitions ni des vocabulaires. Les tableaux `.npy` mappés
sur disque conservent, dans l'ordre demandé des observations :

| Tableau | Forme | Type / sens |
|---|---|---|
| numeric | N × 100 × 6 | float64 : price, bid, ask, bid_size, ask_size, flux |
| venues | N × 100 | int64, codes bruts non réinterprétés |
| categorical | N × 100 × 3 | uint8, dictionnaires fixes action/side/trade |
| observation_ids, labels, source_indices | N | int64, alignement conservé |
| ready | N | bool, marqueurs de reprise |

L'axe 100 conserve exactement l'ordre source. `order_id` n'est pas un canal
prédictif ; sa valeur ne sert pas à trier les événements. Les catégories fixes
viennent du schéma (A/D/U, A/B, False/True), pas de l'observation de la validation.
Les résultats sont ouverts en lecture seule pour les consommateurs.

La provenance inclut taille/mtime de X, empreinte de sélection et du module cache.
Ce n'est pas un SHA-256 de X. Les données sont flushées avant les marqueurs de
complétion lors d'un arrêt contrôlé ; aucune atomicité de panne matérielle n'est
revendiquée. Une reprise relit le préfixe et conserve les lignes déjà complètes.
Un cache complet ne relit pas X. Les bornes de durée sont coopératives.

La complétion certifie les identifiants sélectionnés, pas un nouvel audit global
X/y. Le lecteur assemble et valide les observations rencontrées ; le parseur
peut anticiper un chunk au-delà du dernier événement effectivement consommé.

## Encodage propre au train

`EventEncoder.fit(cache, train_indices)` ajuste StandardScaler sur les six canaux
numériques du train et découvre ses venues uniquement. Par événement : six
valeurs numériques, six slots de venues plus inconnu, trois indicateurs action,
deux side et deux trade donnent **20 canaux float32**. Les codes catégoriels ne
sont pas employés comme valeurs continues. `transform` ne refait aucun fit.

Cet encodage à 20 canaux est un choix de pilote, pas une revendication d'identité
avec la représentation à 30 canaux du benchmark. Une future campagne devra
reconstruire l'encodeur pour chaque train de fold.

## Périmètre fixé avant l'exécution réelle

- Protocole historique : lockbox 20 %, cinq folds stratifiés, graine 42.
- Seulement les 512 premiers indices globaux du train du fold 0, dans l'ordre
  fourni par le découpage ; aucune ligne de validation ou lockbox.
- Lecture bornée à 5000 observations consommées et 300 secondes ; chunks de
  20000 événements, soit une anticipation supplémentaire d'au plus un chunk
  et une observation en cours d'assemblage.
- Deux threads CPU ; huit lots de 32 par réseau, soit 256 exemples vus ;
  120 secondes coopératives maximum par modèle.
- Adam, learning_rate=0,001, seed=42 ; aucune recherche d'hyperparamètres.
- CNN : deux convolutions 16 canaux, noyaux 5 puis 3, ReLU, pooling moyen.
- GRU : une couche bidirectionnelle, 16 unités par direction, états finaux concaténés.
- Sortie : 24 logits, CrossEntropyLoss moyenne. Seules des losses de train sont
  rapportées ; aucune accuracy ou métrique de validation n'est calculée.

## Commandes

PyTorch CPU et psutil sont déjà présents dans l'environnement utilisé. L'extra
optionnel `sequences` déclare ces besoins, sans installer torch pour le seul
chemin tabulaire. Pour une nouvelle installation CPU, choisir explicitement
l'index puis les dépendances PyPI ; aucune installation n'est relancée ici :

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev,sequences]"
```

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/unit/test_sequence_cache.py tests/unit/test_sequence_models.py -q
.\.venv\Scripts\python.exe -u -m cfm_challenge.sequences.pilot --output reports/experiments/sequence-pilot-new --cache data/processed/sequence-pilot-new
```

Choisir un dossier de sortie neuf ; la provenance doit correspondre pour reprendre
un cache existant. Les données ne sont jamais enregistrées dans les rapports.
Un manque de mémoire disponible arrête le pilote aux jalons de contrôle.

## Interprétation et C++

La lecture/assemblage est chronométrée autour de l'avancement du lecteur ; les
conversions sont mesurées séparément. Le chargement/encodage des lots et les
passes réseau (forward, backward, optimizer) ont des durées distinctes.
Ce profil borné ne prédit pas une durée garantie sur tout le développement.

Un noyau C++ ne sera retenu qu'après confirmation du coût sur un volume pertinent.
Une amélioration du cache n'est pas une accélération du réseau ; le CNN et le
GRU utilisent déjà les noyaux compilés de PyTorch. Aucun C++ ni cloud n'est ajouté.

Questions de compréhension : pourquoi l'encodeur dépend-il du train alors que
le cache n'en dépend pas ? Que signifie une loss finie après huit lots ?
Pourquoi une mesure sur 512 observations ne suffit-elle pas à annoncer un gain C++ ?
