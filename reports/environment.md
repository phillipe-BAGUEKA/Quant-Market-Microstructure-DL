Environnement local vérifié le 8 septembre 2026.

- Système : Windows 11 Famille x64, version 10.0.26200, build 26200.
- Python : CPython 3.12.6, 64 bits. Interpréteur de base sélectionné : Python 3.12 global, invoqué avec `py -3.12`.
- Interpréteur utilisé après création : `.\.venv\Scripts\python.exe`.
- `.venv` était absent avant l'opération. Il utilise `include-system-site-packages = false`, sans activation ni changement de politique PowerShell.
- Matériel : Intel Core 7 150U, 10 cœurs physiques / 12 processeurs logiques ; 16 Gio de RAM installée, 15,692 Gio utilisables, environ 2,756 Gio disponibles lors du relevé avant installation.
- Outillage existant : g++ 6.3.0 (MinGW.org) détecté dans le PATH. Visual Studio Community 2022 17.14.14 est recensé ; son composant MSVC x86/x64 ne l'est pas. clang++, cl.exe et CMake non détectés dans le PATH et quelques emplacements usuels, sans recherche exhaustive. Cela n'exclut pas une installation ailleurs.

Versions principales effectivement installées :

| Paquet | Version |
|---|---|
| pip | 26.2.1 |
| torch | 2.14.0+cpu |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |
| ipykernel | 7.3.0 |
| pytest | 9.1.1 |

Commandes d'installation exécutées depuis la racine du dépôt. Les variables ci-dessous étaient limitées aux processus PowerShell d'installation, sans modification persistante de configuration ; `PIP_NO_INDEX=0` permet l'accès aux index explicitement indiqués.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip
$env:PIP_CONFIG_FILE = 'NUL'
$env:PIP_NO_INDEX = '0'
$env:PIP_NO_CACHE_DIR = 'true'
$env:PIP_REQUIRE_VIRTUALENV = 'true'
.\.venv\Scripts\python.exe -m pip install --upgrade pip --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scikit-learn ipykernel pytest --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip freeze --all
```

Le dernier résultat a été enregistré en UTF-8 dans `reports/environment-freeze.txt`, y compris pip et les dépendances transitives. PyTorch stable a été obtenu depuis l'[index CPU officiel](https://download.pytorch.org/whl/cpu), sans nightly, torchvision ni torchaudio ; voir aussi les [instructions officielles PyTorch](https://pytorch.org/get-started/locally/).

Vérifications réussies :

- Python 3.12.6 x64 ; `sys.executable` et `sys.prefix` pointent dans `.venv`, et `sys.prefix != sys.base_prefix`.
- `site.ENABLE_USER_SITE` vaut `False`. Les répertoires de paquets globaux et utilisateur sont absents de `sys.path` ; tous les modules demandés sont importés depuis `.venv\Lib\site-packages`.
- Imports et versions des sept bibliothèques vérifiés ; `pip check` : `No broken requirements found.`
- `torch.version.cuda is None` et `torch.cuda.is_available() is False`. La version installée n'est ni une préversion ni une version de développement.
- Calcul effectué uniquement en ligne de commande, sur CPU : pour x = [1, 2, 3], somme(x²) = 14 ; après rétropropagation, gradient [2, 4, 6], exactement égal au résultat attendu 2x.

Pour recréer ces versions dans un NOUVEL environnement Windows x64 avec Python 3.12.6, reprendre la création et les variables ci-dessus, puis :

```powershell
.\.venv\Scripts\python.exe -m pip install pip==26.2.1 --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r .\reports\environment-freeze.txt --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip check
```

`pip freeze` ne mémorise pas à lui seul l'index CPU, la version de Python ou le système. Il faut installer d'abord la version exacte de torch depuis l'index CPU, puis appliquer le relevé depuis PyPI ; torch est alors déjà satisfait. Ne pas écraser un `.venv` existant.

Incidents résolus : la première création de `.venv` s'est interrompue lors d'ensurepip ; pip a ensuite été initialisé hors du bac à sable avec autorisation et les scripts standard de venv ont été complétés via la bibliothèque standard, sans activation. Les lectures CIM et l'installation ont également nécessité l'exécution autorisée hors du bac à sable. Une tentative pip dans le bac à sable a signalé des échecs de nettoyage de dossiers temporaires ; aucun nettoyage manuel n'a été effectué. Aucun blocage subsistant.

Les données n'ont été ni modifiées ni déplacées. `.venv/` et `data/` sont ignorés. Les règles exactes `__pycache__/` et `**pycache**/` étaient déjà présentes ; `.gitignore` est préservé. Aucun code métier, notebook, modèle ou pipeline créé, aucune installation de compilateur/CUDA/outillage C++, aucune modification d'Anaconda ou de l'installation Python globale. Aucun git add, commit, push ou changement de branche.
