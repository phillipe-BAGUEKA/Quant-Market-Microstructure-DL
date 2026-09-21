# Audit agrégé des splits

- observations totales: 160800
- classes: 24
- développement: 128640
- lockbox: 32160
- blocs candidats: 335
- taille nominale: 480
- taille dernier bloc: 480
- dernier bloc complet: oui
- Python: 3.12.6
- pandas: 3.0.5
- scikit-learn: 1.9.0
- SHA-256 développement: ebb410d7aa64531b59081d7a8317378ad5985b0c165a265e7ddc60a8ea530616
- SHA-256 lockbox: c4ff064eb03770fe856921279424e346ba48fd2c6745b8a067876b11d80d1b18

## Folds
- stratifié 1: train=102912, validation=25728
- stratifié 2: train=102912, validation=25728
- stratifié 3: train=102912, validation=25728
- stratifié 4: train=102912, validation=25728
- stratifié 5: train=102912, validation=25728
- groupé 1: train=102825, validation=25815, groupes séparés=oui
- groupé 2: train=102830, validation=25810, groupes séparés=oui
- groupé 3: train=103050, validation=25590, groupes séparés=oui
- groupé 4: train=102812, validation=25828, groupes séparés=oui
- groupé 5: train=103043, validation=25597, groupes séparés=oui

## Limites

Le lockbox est stratifié et i.i.d., non temporel. Les candidate_block_id sont un test secondaire de sensibilité et ne représentent pas des journées. La métrique officielle du classement n’est pas confirmée localement.
