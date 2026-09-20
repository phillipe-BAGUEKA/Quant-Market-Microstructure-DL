## Contrôle ciblé de correspondance — 8 septembre 2026

Contrôle effectué avec `.\.venv\Scripts\python.exe` (Python 3.12.6) et pandas 3.0.5. Périmètre : uniquement `data/raw/X_train.csv` et `data/raw/y_train.csv`.

**Conclusion agrégée**

Tous les identifiants contrôlés dans `X_train.csv` possèdent exactement une étiquette non manquante dans `y_train.csv`.

**Schéma effectivement observé**

Les en-têtes et les cinq premières lignes de données de chaque fichier ont été lus pour établir le schéma et inférer les types ci-dessous. La clé commune est `obs_id` ; la colonne cible de y est `eqt_code_cat`. Aucun en-tête vide, aucune colonne `Unnamed: ...` ni colonne supplémentaire d'index exporté n'ont été observés. L'index pandas créé en mémoire n'est pas une colonne du CSV. Numéro de ligne, index pandas, `order_id` et `obs_id` ne sont pas interchangeables.

**X_train.csv — 12 colonnes, dans l'ordre du fichier**

| Colonne exacte | Type pandas observé sur cinq lignes |
|---|---|
| `obs_id` | `int64` |
| `venue` | `int64` |
| `order_id` | `int64` |
| `action` | `str` |
| `side` | `str` |
| `price` | `float64` |
| `bid` | `float64` |
| `ask` | `float64` |
| `bid_size` | `int64` |
| `ask_size` | `int64` |
| `trade` | `bool` |
| `flux` | `int64` |

**y_train.csv — 2 colonnes, dans l'ordre du fichier**

| Colonne exacte | Type pandas observé sur cinq lignes |
|---|---|
| `obs_id` | `int64` |
| `eqt_code_cat` | `int64` |

**Correspondance globale des identifiants**

| Contrôle | Nombre de cas |
|---|---:|
| Identifiants distincts non manquants dans X_train | 160 800 |
| Identifiants distincts non manquants dans y_train | 160 800 |
| Identifiants dans X_train mais absents de y_train | 0 |
| Identifiants dans y_train mais absents de X_train | 0 |
| Lignes de y_train avec identifiant manquant | 0 |
| Lignes de y_train avec label manquant | 0 |
| Identifiants dupliqués dans y_train | 0 |
| Doublons de y_train avec labels identiques renseignés | 0 |
| Doublons de y_train avec labels contradictoires | 0 |
| Doublons de y_train mêlant label renseigné et manquant | 0 |
| Doublons de y_train avec tous les labels manquants | 0 |

Aucune anomalie détectée dans ce périmètre ; aucun exemple d'anomalie à présenter.

**Méthode et limites**

Après l'aperçu de cinq lignes, un seul parcours complet de X a été effectué par morceaux de 250 000 lignes, avec `usecols=['obs_id']` et lecture explicite en chaînes pour préserver les représentations. Seuls la colonne identifiant du morceau courant, les ensembles/représentations d'identifiants et les compteurs nécessaires ont été conservés. Aucune seconde lecture complète de X n'a été utilisée. Le petit fichier y a été lu intégralement.

Les identifiants sont inférés `int64` sur les aperçus. Le contrôle complet de leurs textes bruts confirme exclusivement des écritures entières décimales canoniques dans les deux fichiers, sans espace ni variante de représentation ; aucune normalisation n'a changé une clé. Aucun identifiant manquant n'a non plus été rencontré dans X. Les valeurs manquantes ont été recherchées explicitement (cellules vides, espaces seuls et marqueurs NA usuels), sans considérer le nombre 0 comme manquant. Les ensembles d'identifiants ont été comparés sans supposer tri ou alignement des lignes.

Le parcours complet a réussi : la correspondance globale est vérifiée. Les types des autres colonnes ont été inférés uniquement sur les cinq lignes de l'aperçu ; ils ne constituent pas une validation de type de tout X. Aucune signification économique n'est attribuée aux colonnes. Les fichiers de test et l'exemple de soumission n'ont pas été inspectés.

Ce rapport ne contient aucune ligne ni valeur individuelle issue des données brutes.

Les deux fichiers sources ont été ouverts en lecture seule ; tailles et dates de modification sont restées identiques pendant le contrôle. Aucun fichier de données modifié, aucun nettoyage, dédoublonnage ou autre correction, aucun code métier, modèle, pipeline, notebook ou graphique créé. La branche `main` est conservée. Aucun git add, commit ou push effectué.
