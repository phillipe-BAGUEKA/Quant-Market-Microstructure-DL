# Cache ordonné et pilotes CPU — bilan du 3 novembre 2026

Le socle B0/B1 a été publié avant ce jalon dans le commit
`30f3534cfd6e2d640e99d29d660488593146fc7d` sur main. Les dossiers d'exécution
gardent leur préfixe 20261102 choisi au début de la session.

## Travaux terminés et limites

Le cache séquentiel est implémenté et qualifié par des tests synthétiques
(ordre, chunks de 137, alignement, reprise et exclusions) et un petit cache réel
de **512 observations train**, contenant les 24 classes. Aucun lockbox, jeu de
test ou ensemble de validation n'est utilisé pour ces pilotes.

La préparation a consommé **806 observations physiques**, soit 80600 événements.
Le lecteur par chunks peut parser davantage : borne conservative 100700 lignes
avec l'anticipation d'un chunk de 20000 et d'une observation de 100 événements.
Le résultat comporte numeric (512,100,6) float64, venues (512,100) int64 et
categorical (512,100,3) uint8, avec identifiants/labels/indices sources alignés.
Les canaux d'entrée des réseaux ont la forme (batch,100,20) float32.

Le scaler et les venues sont appris sur ces 512 lignes, toutes issues du train
du fold 0. Chaque réseau voit huit lots de 32, soit 256 observations, sans
shuffle. Cela teste la chaîne d'apprentissage ; ce n'est pas une campagne
représentative de la performance des architectures ni la reproduction du benchmark.

## Mesures réelles

| Étape | Durée |
|---|---:|
| Cache de 512 observations, ouverture/contrôles compris | 0,9933 s |
| Lecture et assemblage du préfixe consommé | 0,2908 s |
| Conversion et écriture des séquences sélectionnées | 0,4959 s |
| Ajustement de l'encodeur train | 0,00693 s |

Les deux sous-durées ne couvrent pas tout le temps total : flush, contrôles et
journalisation s'ajoutent. Elles ne prédisent pas le temps du cache complet.

| Réseau pilote | Paramètres | Chargement/encodage des 8 lots | Forward/backward/optimizer | Total des 8 lots |
|---|---:|---:|---:|---:|
| CNN 1D | 2808 | 0,01147 s | 0,02962 s | 0,04759 s |
| GRU bidirectionnel | 4440 | 0,00875 s | 0,22102 s | 0,23617 s |

Le total par réseau débute après sa construction et celle de l'optimiseur.
L'ensemble du pilote journalisé dure 4,1064 s, hors imports initiaux ; la
commande shell a terminé en environ 7,79 s, code 0. Pic résident global :
**409,60 Mio**. Le processus est partagé : ce pic n'isole pas la mémoire propre
à chaque architecture. Deux threads CPU, Adam 0,001, seed 42, loss non pondérée.

Sorties (32,24), losses et gradients finis, paramètres modifiés après les pas
d'optimisation : contrôlés pour chaque modèle. Les losses de train des huit
lots sont archivées dans [CNN1D.json](20261102-sequence-pilot/CNN1D.json) et
[BiGRU.json](20261102-sequence-pilot/BiGRU.json). Elles fluctuent entre lots ;
aucune baisse monotone ni convergence n'est revendiquée. Aucun score de
validation n'a été calculé ou comparé aux métriques B1.

Le [manifeste](20261102-sequence-pilot/manifest.json) contient la configuration,
les empreintes du code non commité, de y et de la sélection. Les données et
identifiants individuels restent dans le cache ignoré, pas dans ce rapport.

## Cache complet non construit

Deux contrôles préalables ont refusé de démarrer la préparation de 128640
observations : RAM disponible inférieure au seuil initial de 1 Gio. Lors du
second, 952,71875 Mio étaient disponibles après chargement de y. Le seuil n'a
pas été abaissé pour forcer le démarrage. **Aucun scan complet de X n'a été
lancé et aucun cache complet n'est annoncé comme validé.**

Le stockage des valeurs complet représenterait environ 724 Mio plus indices et
métadonnées. Des tableaux mappés limitent l'allocation Python, sans garantir
l'absence de pression mémoire ou de pagination. Recommander 1,5 à 2 Gio de RAM
disponible avant de relancer le parcours borné à 600 s, avec arrêt sous 384 Mio.

## Budget proposé et C++

Après construction du cache complet, mesurer d'abord une seule époque du fold 0
pour chacun des deux petits réseaux, avec plafond de 10 minutes par modèle,
mêmes lots de 32 et deux threads. C'est une proposition, pas une expérience
effectuée. L'extrapolation mécanique des huit lots donne environ 17 s CNN et
92 s GRU pour 3216 lots train ; ces nombres ignorent cache froid, pagination,
validation et variations de charge, et ne sont pas des durées garanties.

Sur ce préfixe, la conversion des séquences coûte plus que lecture/assemblage.
Le candidat commun à profiler à plus grande échelle est la conversion/validation
par lots des canaux et catégories vers le stockage contigu. Commencer par
mesurer une version vectorisée Python/NumPy avant un noyau C++ équivalent.
Le coût absolu actuel est trop petit et l'échantillon trop limité pour justifier
une réécriture du lecteur ou annoncer un gain C++. Les réseaux utilisent déjà
les noyaux compilés PyTorch. Aucun C++, compilation, GPU ou cloud engagé.

## Vérifications et statut de publication

- Tests ciblés : 10 passed in 9.24s, code 0.
- Suite complète : 284 passed in 8.95s, code 0.
- Formateur isolé : Black 25.1.0, sans changement de l'environnement projet.
- [Commandes exactes](20261103-sequence-commands.md), dont les échecs mémoire conservés.
- Une seule revue indépendante pendant l'intervention, dédiée au commit B0/B1.
  Le jalon séquentiel a été relu et testé par l'agent principal ; aucune seconde
  revue indépendante n'est revendiquée.

Le [guide](../../docs/sequences.md) décrit les API, dtypes, reprise et encodage.
Les installations documentées et entraînements complets ne sont pas exécutés.

SEQUENCE_CACHE_SYNTHETIC_VALIDATED=true
SEQUENCE_CACHE_REAL_512_VALIDATED=true
SEQUENCE_CACHE_FULL_DEVELOPMENT_VALIDATED=false
CNN_CPU_PILOT_VALIDATED=true
BIDIRECTIONAL_GRU_CPU_PILOT_VALIDATED=true
SEQUENCE_FIVE_FOLD_EVALUATION=false
