# Cache ordonné et expériences séquentielles CPU

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

## Pilote historique sur 512 observations

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

## Lanceur d'une époque complète sur le fold 0

Le [bilan du cache](../reports/experiments/20261104-sequence-results.md) conserve
la première interruption mémoire. La [reprise terminée](../reports/experiments/20261104-sequence-retry-results.md)
documente désormais une époque et une validation complète du fold 0 pour les
deux réseaux, ainsi que la veille Windows qui avait interrompu le BiGRU auparavant.
Ce n'est pas une campagne à cinq folds.

`sequences.experiment` utilise les architectures `PilotCNN` et `PilotBiGRU`
existantes. Il exige un cache de développement **déjà complet** : cette commande
ne prépare pas X et refuse un checkpoint partiel avant d'avancer le lecteur.

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
@'
from pathlib import Path
import psutil
from cfm_challenge.sequences.experiment import run

if psutil.virtual_memory().available < 2560 * 2**20:
    raise RuntimeError("Au moins 2,5 Gio disponibles après imports sont requis")
run(
    Path("reports/experiments/sequence-fold0-new"),
    Path("data/processed/20261104-sequences-development"),
)
'@ | .\.venv\Scripts\python.exe -u -
```

Cette enveloppe contrôle les **2,5 Gio disponibles après imports** ; la CLI
seule conserve seulement le garde de 384 Mio. Les [commandes réellement exécutées](../reports/experiments/20261104-sequence-retry-commands.md)
ajoutent la vérification des empreintes et une interdiction d'ouvrir X/test.
Garder le capot ouvert : une veille est comptée dans le budget de temps écoulé.

Le dossier de résultat doit être neuf. Le protocole reste 20 % de lockbox,
cinq folds stratifiés, graine 42 ; seul le fold 0 est exécuté. Les indices
globaux sont localisés dans le cache des 128 640 observations. L'encodeur est
ajusté sur les 102 912 lignes du train (10 291 200 événements), puis conservé
inchangé pour les 25 728 lignes de validation et pour les deux réseaux.

| Paramètre fixé avant exécution | Valeur |
|---|---|
| Époques / modèle | 1 |
| Optimiseur | Adam, learning rate 0,001 ; valeurs effectives archivées |
| Taille des lots | 32, dernier lot conservé |
| Ordre du train | permutation NumPy déterministe, graine 42 |
| CPU | deux threads PyTorch, algorithmes déterministes |
| RAM initiale | au moins 2,5 Gio après imports |
| Budget par modèle | 600 s : initialisation, train, validation et métriques |
| Arrêt mémoire | RAM disponible sous 384 Mio |

Le plafond est coopératif, contrôlé entre les lots et après les phases ; il
peut être dépassé du temps d'une opération en cours. En cas d'arrêt, le JSON
du modèle conserve l'étape et les nombres de lignes et lots terminés. Aucun
résultat partiel n'est présenté comme une validation complète. Les poids ne
sont pas archivés : les journaux permettent de reproduire l'expérience, pas
de reprendre un apprentissage interrompu. La reprise concerne le cache.

La validation s'exécute en `eval()` et `no_grad()`. La cross-entropy est une
somme de pertes calculées par log-softmax float64, divisée par le nombre total
d'observations ; elle ne moyenne pas à poids égal des lots de tailles différentes.
La log-loss, l'accuracy, la macro-F1 et la confusion 24 × 24 utilisent les métriques
tabulaires existantes, dans l'ordre des classes 0 à 23. Les éventuels clips de
probabilités par scikit-learn sont comptés. La loss moyenne du train est celle
observée au fil des mises à jour, pas une seconde évaluation du train final.

Le manifeste fixe paramètres, versions, empreintes du code, de y et des indices
de développement. Les journaux séparent ouverture/contrôle du cache, ajustement
de l'encodeur, chargement/encodage des lots, apprentissage, validation et mémoire.
Le pic mémoire du processus est cumulatif : il n'est pas remis à zéro entre
CNN et GRU. Une époque sur un fold ne classe pas définitivement les architectures.

Sur le run terminé : CNN 30,90 s, BiGRU 245,66 s, encodeur 3,62 s, pic RSS
1 195 Mio. Le minimum disponible pendant l'exécution est de 1 954 Mio.
Les cinq époques proposées pour la suite ne sont ni implémentées dans ce
lanceur ni exécutées ; leur budget estimatif reste une décision à prendre.

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
