# Reprise séquentielle du fold 0 — 4 novembre 2026

## Résultats acquis

Le CNN 1D et le BiGRU ont chacun terminé **une époque complète et leur validation
sur le fold 0**, dans la troisième commande de reprise. Le code de sortie final
est **0**. Les tentatives interrompues ou refusées auparavant sont conservées
séparément ; elles ne sont pas présentées comme des validations réussies.

| Modèle, validation du fold 0 | Accuracy | Macro-F1 | Cross-entropy | Log-loss |
|---|---:|---:|---:|---:|
| B1 historique, convergée | 0,2793454602 | 0,2681332673 | Non enregistrée séparément | 2,2865579895 |
| CNN 1D, une époque | 0,1869558458 | 0,1348607638 | 2,5196802320 | 2,5193273550 |
| BiGRU, une époque | 0,1852845149 | 0,1533988132 | 2,4873520367 | 2,4873520367 |

B1 reste devant les deux réseaux sur ce fold : écart d'accuracy de **9,239 points**
avec le CNN et **9,406 points** avec le BiGRU.
Ce n'est pas un classement définitif des architectures : B1 a convergé,
tandis que les réseaux n'ont reçu qu'une époque, sans recherche d'hyperparamètres.
Le BiGRU obtient une macro-F1 et une perte meilleures que le CNN, mais une
accuracy légèrement inférieure et un temps nettement supérieur.

La cross-entropy est calculée directement sur les logits par log-softmax en
float64. La log-loss scikit-learn tronque les probabilités à epsilon float64.
Le CNN présente 146 probabilités sous ce seuil ; l'écart de **0,0003528770**
entre les deux pertes ne relève donc pas d'un simple arrondi. Il n'y a ni
pondération des classes ni lissage des labels. Pour le BiGRU, aucune probabilité
n'a été tronquée et les deux pertes coïncident dans le résultat enregistré.

Les [deux matrices de confusion complètes](20261104-sequence-confusions.md)
sont aussi archivées dans le champ `confusion_matrix` des résultats
[CNN](20261104-sequence-fold0-retry-3/CNN1D.json) et
[BiGRU](20261104-sequence-fold0-retry-3/BiGRU.json), de forme **24 × 24** :
lignes = classes réelles, colonnes = classes prédites, ordre 0 à 23.
Chaque ligne somme à **1 072**, soit **25 728 observations**. Accuracy et
macro-F1 ont été recalculées indépendamment depuis cette matrice et correspondent
aux valeurs archivées à une tolérance absolue de 1e-15. Les métriques, la loss
moyenne du train et la matrice CNN reproduisent exactement celles de la première
reprise, malgré des durées différentes : les deux CNN sont repartis de zéro.

## Données, initialisation et contrôles

- État initial : main, HEAD = origin/main =
  `e646ae0aae8ec7faee8983977e926d94ccd0be0d`, working tree propre.
- RAM mesurée initialement : **3 003,67 Mio**. Pour l'exécution finale, après
  imports et libération de mémoire par l'utilisateur : **2 937,56 puis 2 992,68 Mio**,
  à dix secondes d'intervalle, au-dessus du seuil initial de **2 560 Mio**.
  L'attente préalable a duré 170,023 s et n'entre pas dans le budget des modèles.
- Les SHA-256 des huit fichiers du cache qualifié correspondent aux empreintes
  archivées ; vérification finale préalable en **1,803 s**, sans lecture des événements X.
- Les sources, tests et pyproject correspondent exactement aux empreintes de
  la tentative précédente. **Aucun code de production ou de test n'est modifié.**
- Un audit Python interdit les ouvertures de X_train, X_test et y_random.
  Le lanceur consulte seulement les métadonnées de X pour sa provenance.
- Développement : **128 640 observations** ; train du fold 0 : **102 912** ;
  validation : **25 728**. Lockbox : 32 160, exclu des ajustements et évaluations.
- Fraction lockbox 0,2, cinq folds, graine 42 ; mêmes empreintes de y et de
  sélection que B1. Indices globaux/localisés et labels contrôlés par le lanceur.
- Cache rouvert et contrôlé en **0,675 s** ; aucun cache reconstruit.
- Encodeur ajusté en **3,621 s**, uniquement sur **10 291 200 événements train** ;
  six venues connues, état inchangé après chacune des deux validations.
- Réinitialisation de chaque réseau avec torch seed 42 ; Adam 0,001, batch 32,
  permutation NumPy seed 42, deux threads CPU, algorithmes déterministes.
- Plafond coopératif : **600 s par modèle**, initialisation, train, validation
  et métriques compris. Le garde mémoire demeure à **384 Mio disponibles**.
- Aucun poids n'est chargé ou sauvegardé. Les rapports conservent les paramètres
  et résultats, pas un checkpoint permettant de poursuivre un train interrompu.

[Manifeste de l'exécution finale](20261104-sequence-fold0-retry-3/manifest.json) ;
[précontrôle](20261104-sequence-fold0-retry-3/preflight.json) ;
[journal complet](20261104-sequence-fold0-retry-3/progress.jsonl) ;
[commandes exactes](20261104-sequence-retry-commands.md).

## Durées et mémoire

| Phase | CNN 1D terminé | BiGRU terminé |
|---|---:|---:|
| Initialisation | 0,010 s | 0,003 s |
| Chargement et encodage train | 5,593 s | 5,583 s |
| Forward/backward/optimiseur | 19,543 s | 223,389 s |
| Train, durée totale | 26,626 s | 230,592 s |
| Chargement et encodage validation | 1,755 s | 1,623 s |
| Forward validation | 1,931 s | 12,812 s |
| Calcul final des métriques | 0,138 s | 0,185 s |
| Validation, durée totale | 4,244 s | 15,050 s |
| Total modèle | **30,896 s** | **245,658 s** |
| Lignes train traitées | 102 912 | 102 912 |
| Lots train terminés | 3 216 | 3 216 |
| Lignes validation traitées | 25 728 | 25 728 |
| Lots validation terminés | 804 | 804 |

Le pic RSS du processus final est **1 195,09 Mio**, cumulatif et non remis à zéro
entre les deux modèles (1 185,52 Mio après CNN). La plus faible disponibilité
journalisée pendant le run est **1 954,48 Mio**. Les deux modèles sont sous
leur plafond de 600 secondes et aucun garde n'interrompt cette exécution.
Durée complète PowerShell, attente mémoire et imports inclus : **470,982 s** ; sortie **0**.
Durée du lanceur après précontrôle : **287,580 s**.
Les totaux incluent les gardes, journaux et opérations non attribuées aux
sous-phases ; il ne faut pas additionner un total et ses composantes.

## Historique : diagnostic de la première interruption BiGRU

La première reprise de cette intervention avait terminé le CNN en 27,389 s,
puis arrêté le BiGRU à **20 352 lignes / 636 lots de train**, avec zéro ligne
de validation. Cette commande a duré 693,978 s et s'est terminée avec code 1.
Le minimum de RAM journalisé était de 1 382,52 Mio, au-dessus du garde mémoire.

Windows confirme une veille moderne due au **capot** :

| Événement Kernel-Power | Heure Europe/Kiev |
|---|---|
| 506, entrée en veille, motif Lid | 09:11:40,306 |
| 507, sortie de veille, motif Lid | 09:21:39,529 |

Cette veille a duré **599,223 s**. Le dernier jalon avant la rupture comptait
600 lots et 37,241 s dans la section de calcul. Après 636 lots, cette section
affiche 639,916 s. La rupture coïncide avec la veille de Windows, pas avec
600 secondes démontrées de calcul CPU actif.

Le contrôle du budget intervient après le lot en cours : il a constaté
**641,320 s** au retour, soit 41,320 s au-delà du seuil. Le budget n'a pas été
allongé, et aucune durée corrigée après coup ne sert à valider le modèle.

[Diagnostic archivé](20261104-sequence-fold0-retry/interruption-diagnosis.json) ;
[résultat interrompu](20261104-sequence-fold0-retry/BiGRU.json).

Une seconde commande a ensuite été refusée **avant entraînement** :
2 223,89 Mio après imports, seuil requis 2 560 Mio, sortie 1, durée 8,405 s.
Son [précontrôle](20261104-sequence-retry-2-preflight.json) est conservé.
Aucune application n'a été fermée automatiquement et aucune politique
d'alimentation n'a été modifiée.

Après confirmation de l'utilisateur, la troisième commande a attendu deux
mesures au-dessus de 2,5 Gio avant de réexécuter le lanceur existant, donc le CNN
puis le BiGRU, tous deux réinitialisés. Le capot est resté ouvert et le run a
terminé. Ce diagnostic explique cette interruption précise ; il ne résout pas
à lui seul les écarts historiques de préparation tabulaire de 18 et 96 minutes.

## Vérifications et limites

Les **14 tests séquentiels existants passent en 18,45 s**, code 0. Aucune nouvelle
fonctionnalité n'est introduite ; la suite unitaire complète n'est pas relancée
dans cette reprise, le code étant inchangé (288 tests réussis lors de sa qualification).

Les 21 fichiers de cette intervention et 49 liens locaux ont été contrôlés,
ainsi que le parsing JSON/JSONL et l'espacement des fichiers non suivis.
La revue indépendante n'a pas trouvé d'erreur dans les métriques ou leur
provenance. Son constat sur les accents dégradés du document de matrices a
été retenu et corrigé ; les tableaux ont ensuite été comparés aux JSON.
Aucun constat n'a été écarté. `git diff --check` ne relève aucune erreur,
seulement les avertissements habituels LF/CRLF.

Les deux réseaux disposent désormais d'un résultat sur **un seul fold, une seule
graine, une seule époque**. Le lockbox et le test restent réservés. Les caches et les
données demeurent locaux ; seuls les rapports agrégés et la documentation sont
destinés à la publication. Les premiers rapports, y compris les échecs, ne sont
ni remplacés ni réattribués à un autre code.

Les [contrôles après exécution](20261104-sequence-fold0-retry-3/checks.json)
confirment supports, métriques issues des matrices, concordance du CNN,
protocole identique à B1, code inchangé et empreintes du cache inchangées après
les exécutions. Aucun résultat séquentiel à cinq folds ni convergence des
réseaux n'est revendiqué.

## Budget proposé, non lancé

Pour observer les courbes d'apprentissage avant de choisir une architecture :
**cinq époques au total pour chacun des deux réseaux**, sur le même fold 0,
depuis une initialisation reproductible. Il faudra un lanceur multi-époques
et des checkpoints ; le lanceur actuel est limité à une époque et ne conserve
pas les poids.

| Réseau | Projection simple de cinq époques, validation à chaque époque | Budget prudent proposé |
|---|---:|---:|
| CNN 1D | 154,5 s, soit 2,6 min | 5 min |
| BiGRU | 1 228,3 s, soit 20,5 min | 30 min |
| Total | Environ 23 min | **35 min en local** |

Cette projection multiplie les durées mesurées par cinq ; elle n'est pas une
garantie. Conserver au moins 2,5 Gio avant lancement, le garde mémoire et le
capot ouvert. Le choix de poursuivre les deux réseaux, de privilégier le CNN
ou d'aborder MiniRocket reste à discuter. L'utilisateur a demandé d'abord un
état des lieux : **aucun budget supplémentaire ni choix de modèle n'est validé**.
