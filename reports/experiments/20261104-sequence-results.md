# Cache complet et première époque séquentielle — 4 novembre 2026

## Préparation terminée

Le cache local contient **128 640 observations de développement**, chacune avec
100 événements : 12 864 000 événements conservés. Les 32 160 observations du
lockbox sont exclues du stockage, du fit de l'encodeur et de l'évaluation.

Le protocole reste `SplitConfiguration()` : fraction 0,2, cinq folds, graine 42.
Le fold 0 contient 102 912 observations de train et 25 728 de validation.
Les empreintes de y et des indices de développement sont identiques à celles
de [B1](20261102-tabular-final-2000/manifest.json).

| Mesure de préparation | Résultat |
|---|---:|
| RAM disponible, première mesure psutil | 1 972,44 Mio |
| RAM disponible après sélection, avant X | 2 416,07 Mio |
| Seuil initial / seuil d'arrêt | 1 024 / 384 Mio |
| Construction du cache | 385,695 s |
| Lecture et assemblage | 112,443 s |
| Conversion et écriture des lignes | 261,706 s |
| Temps total après imports, contrôles et empreintes compris | 392,723 s |
| Pic RSS du processus | 972,50 Mio |
| Minimum disponible aux points de contrôle | 467,23 Mio |
| Stockage local | 762 193 318 octets, soit 726,88 Mio |

Un seul parcours a consommé les 160 800 observations de X. Le lecteur rencontre
physiquement les lignes du lockbox intercalées, mais elles ne sont pas mises en
cache. Aucun jeu de test n'est ouvert. La différence entre durée totale,
lecture et conversions inclut flush, journaux et contrôles ; elle n'est pas
attribuée à un goulot sans mesure supplémentaire.

Les contrôles portent sur toutes les lignes du cache : formes et types,
finitude, bornes des catégories, égalité des identifiants/labels/indices à la
sélection et absence du lockbox. Le lecteur exige 100 événements et conserve
l'ordre source ; la copie du cache ne trie rien. L'exactitude de cette copie
ordonnée est couverte par les tests synthétiques. Il n'y a pas de deuxième
comparaison exhaustive entre CSV et cache, ni de nouvel audit global indépendant.

Les empreintes SHA-256 des huit fichiers du cache sont archivées dans
[result.json](20261104-sequence-cache/result.json). La provenance de X utilise
sa taille et sa date de modification, pas un SHA-256 du CSV complet. Le cache
est ignoré par Git ; seuls les rapports agrégés sont publiés. Les interruptions
contrôlées conservent les marqueurs de reprise ; aucune atomicité face à une
panne matérielle n'est revendiquée.

## Première tentative d'époque : arrêt mémoire

[Manifeste](20261104-sequence-fold0-epoch/manifest.json),
[journal](20261104-sequence-fold0-epoch/progress.jsonl) et
[résultat CNN interrompu](20261104-sequence-fold0-epoch/CNN1D.json).

Configuration fixée avant le run : architectures existantes, une époque,
Adam 0,001, lots de 32, permutation train graine 42, deux threads CPU,
algorithmes déterministes, maximum 600 s par modèle pour initialisation,
apprentissage, validation et métriques. Ordre des classes 0 à 23.

Le cache a été rouvert et contrôlé en 1,593 s, sans nouveau scan X.
L'encodeur a été ajusté en 3,876 s sur **10 291 200 événements du train**,
avec six venues connues. La validation n'intervient dans aucun ajustement.

| Étape CNN | Traité / prévu |
|---|---:|
| Train | 82 688 / 102 912 observations |
| Lots train | 2 584 / 3 216 |
| Validation | 0 / 25 728 |
| BiGRU | Non démarré |

Le garde a arrêté le CNN à **382,95 Mio disponibles**, après 27,272 s :
5,389 s de chargement/encodage et 20,455 s de calcul d'apprentissage.
Pic RSS : 1 111,09 Mio. Commande complète : 46,096 s, **code de sortie 1**.
Aucune accuracy, macro-F1, log-loss de validation ou matrice de confusion
n'est disponible pour cette tentative. La loss de train partielle archivée
n'est pas comparable à la log-loss de validation de B1.

Après l'arrêt : 1 305,33 Mio disponibles. Les principaux ensembles de processus
utilisateur observés sont VS Code (2 847,8 Mio cumulés), Edge (2 512,8 Mio),
Word (350,7 Mio), JetBrains Toolbox (236,6 Mio) et Discord (212,8 Mio).
Ces sommes de working sets peuvent compter des pages partagées plusieurs fois :
elles ne prédisent pas exactement le gain d'une fermeture.
Fermer manuellement des onglets Edge ou fenêtres VS Code inutiles, et sauvegarder
puis fermer Word, peut libérer de la marge. Aucun processus n'a été arrêté
automatiquement et le seuil mémoire n'a pas été abaissé.

## Comparaison et prochaine exécution

| Validation fold 0 | Accuracy | Macro-F1 | Cross-entropy / log-loss |
|---|---:|---:|---:|
| B1 historique, convergée | 0,2793454602 | 0,2681332673 | 2,2865579895 |
| CNN 1D, tentative interrompue | Non mesurée | Non mesurée | Non mesurée |
| BiGRU | Non exécuté | Non exécuté | Non exécuté |

La comparaison demandée reste incomplète. Aucun classement d'architectures ne
peut être tiré de cet arrêt, ni d'une seule époque lorsqu'elle sera terminée.

Recommandation : viser **2 560 Mio disponibles** avant une nouvelle tentative,
soit environ **1,1 à 1,3 Gio supplémentaires** par rapport aux mesures après
l'arrêt. Le pic observé du processus est de 1 111 Mio ; cette cible conserve
une marge pour les variations des autres applications sans abaisser le garde.
Réutiliser le cache existant et choisir un nouveau dossier de rapport.

Le CNN a traité 82 688 observations en 27,27 s. Une simple mise à l'échelle donne
environ **34 s pour le train complet**, hors validation, et sans garantie face
à la pression mémoire. Réserver **deux minutes pour le CNN** est une première
enveloppe pratique, tout en conservant le plafond déjà fixé de dix minutes.
Aucune nouvelle durée d'époque BiGRU n'a été mesurée : conserver son plafond
de **dix minutes** et le mesurer ensuite. Ne pas lancer les cinq folds ou
augmenter le nombre d'époques avant ces deux validations complètes.

Le profil du cache place 261,71 s dans les conversions/écritures contre 112,44 s
dans la lecture/assemblage. Il ne justifie pas encore un changement C++ : le
cache complet évite désormais ce coût lors des entraînements suivants. Aucun
nouveau parcours X, aucune dépense cloud et aucun changement de modèle ne sont
nécessaires à la reprise proposée.

## Code, vérifications et limites

- `sequences/cache.py` et `sequences/models.py` sont inchangés.
- Nouveau `sequences/experiment.py` : cache complet obligatoire, protocole
  reproduit, encodeur train-only, une époque bornée, validation et archivage.
- Nouveau `test_sequence_experiment.py` : couverture exacte des lignes,
  loss pondérée par les effectifs (dernier lot incomplet), arrêt avec compteurs,
  refus d'un cache partiel avant X et fit de l'encodeur uniquement sur le train.
- Quatre tests nouveaux : **4 passed in 10.60s**.
- Tests séquentiels réunis : **14 passed in 8.60s**.
- Suite unitaire complète : **288 passed in 16.92s**.
- Import/CLI `--help`, Black sur les deux nouveaux fichiers : succès.
- Trente liens locaux vérifiés ; tous les nouveaux JSON/JSONL sont lisibles,
  le code exécuté correspond toujours aux empreintes du manifeste. Contrôle
  séparé de l'espacement des fichiers non suivis, puis `git diff --check` :
  succès. Les avertissements LF/CRLF n'indiquent pas une erreur de contenu.
- Les premiers essais de tests dans le bac à sable ont rencontré des refus
  d'accès aux dossiers temporaires ; la relance autorisée a réussi.
- Une revue indépendante en lecture seule n'a identifié aucun blocage. Elle
  a rappelé le budget coopératif et la provenance sans SHA complet de X ;
  aucune correction de production n'a été nécessaire. La variable inutilisée
  dans la commande de préparation est conservée pour reproduire exactement
  la commande exécutée.

Les [commandes exactes](20261104-sequence-commands.md) et le
[guide séquentiel](../../docs/sequences.md) complètent ce rapport.
B1 n'a pas été réentraînée et aucun résultat historique n'a été modifié.
La RAM a ensuite fluctué entre environ 1,6 et 2,2 Gio pendant la documentation,
sans atteindre durablement la cible de 2,5 Gio : aucune relance automatique
du train n'a été faite dans ces conditions.

Questions de compréhension :

1. Pourquoi peut-on partager le cache brut entre modèles, mais ajuster le scaler
   uniquement sur les lignes train du fold ?
2. Pourquoi la loss obtenue pendant un train interrompu ne constitue-t-elle pas
   une mesure comparable au score B1 sur la validation ?
3. Pourquoi le cache sur disque peut-il augmenter le working set sans charger
   explicitement toutes les séquences dans un unique tableau en RAM ?
