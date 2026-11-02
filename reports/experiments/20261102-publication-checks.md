# Contrôles de publication B0/B1

Date : 2 novembre 2026. Branche initiale main ; HEAD et main distant vérifié par
`git ls-remote` : `1e1fc35fabeb84713ba714b283ab5617c76f911f`.
Les travaux B0/B1 étaient encore non commités ; ils sont inclus intégralement.

| Contrôle | Résultat |
|---|---|
| Suite `python -m pytest -p no:cacheprovider tests/unit -q --tb=short` | Code 0, 274 passed in 12.11s |
| Première tentative de tests dans le bac à sable | Code 1, 185 passed / 89 erreurs de permissions sur fixtures temporaires ; relance autorisée ci-dessus |
| Black 25.1.0 `--check` sur baselines et leurs quatre fichiers de tests | Code 0, 11 fichiers inchangés |
| AST avant/après | Identiques hors docstring de module explicitement corrigée ; certificat séparé |
| Compilation en mémoire de src/tests et imports | Code 0 |
| `--help` experiment, audit, profiling, split_audit | Code 0 pour chaque commande |
| Liens Markdown locaux | 21 liens vérifiés, code 0 |
| TOML et espacement des fichiers non suivis | Code 0 |
| `git diff --check` | Code 0 ; avertissements LF/CRLF seulement |
| Exclusions | Quatre CSV, cache et .venv ignorés |
| Recherche ciblée de secrets et chemins personnels | Aucun résultat dans les fichiers destinés au jalon |

Tous les appels Python utilisent `.\.venv\Scripts\python.exe` avec
`PYTHONDONTWRITEBYTECODE=1`. Black est installé uniquement dans un répertoire
temporaire, pas dans l'environnement du projet ni dans Python global.
Aucun linter n'est configuré : aucune prétention de contrôle lint non exécuté.

La revue indépendante unique ne relève aucun problème bloquant ou important.
Sa suggestion esthétique d'ordre des imports locaux n'est pas retenue pour
éviter d'élargir la passe ; les groupes ont été séparés sans changer leur ordre.

Les fichiers historiques d'expérience, leurs métriques et empreintes ne sont
pas modifiés. Leurs JSON/JSONL ont été relus. Les installations documentées,
les scans X, entraînements historiques et liens Web externes ne sont pas
réexécutés comme contrôles courts. Aucun lockbox ou jeu de test évalué.

Le premier contrôle du diff indexé a détecté une ligne vide supplémentaire à
la fin du rapport tabulaire historique, invisible au diff précédent des fichiers
non suivis. Seule cette ligne vide a été retirée ; aucune date, métrique,
configuration ou empreinte historique n'a été changée.
