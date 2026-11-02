# Publication du jalon B0/B1

Cette note décrit la mise en cohérence du 2 novembre 2026. Les rapports
historiques conservent leurs résultats, dates, versions, paramètres et empreintes.

## Formatage et provenance

Les onze modules de baselines et fichiers de tests concernés sont aérés avec
Black 25.1.0, depuis un dossier temporaire isolé. Aucun formateur ou linter
n'était configuré dans le projet. Aucune dépendance runtime n'est ajoutée.
Les suites compactes et appels longs sont dépliés sans changer les calculs.

Le [certificat de formatage](../reports/experiments/20261102-formatting-provenance.json)
conserve les empreintes avant/après et constate l'identité des arbres syntaxiques
Python (hors correction explicite de la docstring du module dataset, devenue
obsolète). Les tests sont relancés ; les entraînements historiques ne le sont pas.
Leurs manifestes continuent donc de désigner les anciennes empreintes.

Les caches vérifient strictement leur provenance. Ne pas modifier leur manifeste
pour contourner cette garde : choisir un nouveau cache pour une nouvelle exécution.

## Documents actuels et historiques

- [README](../README.md) : état actuel et commandes.
- [Résultats B0/B1](../reports/experiments/20261102-tabular-results.md) : expériences achevées.
- [Manifeste original](../reports/experiments/20261102-tabular-final-2000/manifest.json) : configuration exacte.
- [Protocole](validation_protocol.md) : développement, folds, lockbox et métriques.

Les anciens rapports d'environnement, de profilage et d'inspection décrivent leur
exécution d'origine. Ils ne sont pas des inventaires actuels des fonctionnalités.
La décision pédagogique a évolué : l'étude de TCN, InceptionTime et MiniRocket
se déroule en parallèle du développement et n'est plus un prérequis.

## Contrôles et suite

Contrôles courts prévus pour publier : tests synthétiques, imports, `--help`,
liens locaux, TOML, espacement et contenu indexé. Aucun scan B0/B1 ni entraînement
historique n'est relancé. Données brutes, caches et environnements restent ignorés.

Le jalon séquentiel commence après la publication du socle tabulaire ; ses preuves
sont séparées. Un pilote borné ne constitue pas une évaluation complète à cinq folds.
