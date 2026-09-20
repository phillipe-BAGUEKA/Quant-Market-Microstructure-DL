# Profil agrégé de X_train

Source sémantique : [ChallengeData 146](https://challengedata.ens.fr/challenges/146). Ce rapport contient uniquement des agrégats, jamais une ligne, un identifiant ou une valeur d'enregistrement brut.

## Règles officielles

Le contrat sémantique officiel et le benchmark de référence sont paraphrasés dans `docs/challenge_data_contract.md`. Les anomalies ne sont ni corrigées, ni supprimées, ni tronquées.

## Parcours

- Événements : 16080000
- Observations : 160800
- Chunks : 161
- Durée mesurée : 106.761 s

## Constats agrégés

### Types observés

```text
action: str
ask: float64
ask_size: int64
bid: float64
bid_size: int64
flux: int64
obs_id: int64
order_id: int64
price: float64
side: str
trade: bool
venue: int64
```

### Valeurs manquantes

```text
action: 0
ask: 0
ask_size: 0
bid: 0
bid_size: 0
flux: 0
obs_id: 0
order_id: 0
price: 0
side: 0
trade: 0
venue: 0
```

### Valeurs non finies

```text
ask: 0
ask_size: 0
bid: 0
bid_size: 0
flux: 0
price: 0
```

### Plages numériques

```text
ask: (-380.59, 267.94)
ask_size: (1.0, 150115.0)
bid: (-0.7600000000000193, 16.17)
bid_size: (-2.0, 87663.0)
flux: (-50000.0, 31300.0)
price: (-569.77, 199976.33)
```

### Cardinalités

```text
action: 3
ask: 3234
ask_size: 9534
bid: 610
bid_size: 9243
flux: 2413
obs_id: 160800
order_id: 100
price: 41740
side: 2
trade: 2
venue: 6
```

### Modalités

```text
action: {'A': 7972580, 'D': 7966194, 'U': 141226}
side: {'A': 8023288, 'B': 8056712}
trade: {'False': 16003115, 'True': 76885}
venue: {'0': 1823137, '1': 1470971, '2': 1555933, '3': 1374762, '4': 6156289, '5': 3698908}
```

### Signe de flux

```text
negative: 8107420
positive: 7972434
zero: 146
```

### Spread

```text
maximum: 267.94
mean: 0.5946558115671641
minimum: -380.59999999999997
negative: 91652
positive: 15767381
zero: 220967
```

### Tailles négatives

```text
ask_size: 0
bid_size: 8
```

### Premier bid

```text
zero: 160800
```

### Premier order_id

```text
zero: 160800
```

### Répétitions order_id

```text
observations_with_repetition: 160719
observations_without_repetition: 81
repeated_occurrences: 4932076
```

## Analyses conditionnelles

Les tableaux suivants restent agrégés; ils servent à formuler des hypothèses, pas à expliquer définitivement les écarts.

### crossed_by_action

```text
A: 45393
D: 45114
U: 1145
```

### crossed_by_side

```text
A: 46013
B: 45639
```

### crossed_by_trade

```text
False: 91103
True: 549
```

### crossed_by_venue

```text
0: 10745
1: 7554
2: 6950
3: 7788
4: 35847
5: 22768
```

### negative_bid_size_by_action

```text
A: 5
D: 2
U: 1
```

### negative_bid_size_by_side

```text
A: 5
B: 3
```

### negative_bid_size_by_trade

```text
False: 8
```

### flux_by_action

```text
A:positive: 7972434
A:zero: 146
D:negative: 7966194
U:negative: 141226
```

### flux_by_trade

```text
False:negative: 8030535
False:positive: 7972434
False:zero: 146
True:negative: 76885
```

### price_range_by_action

```text
A:max: 199976.33
A:min: -569.77
D:max: 199975.00000000003
D:min: -557.4200000000001
U:max: 265.77000000000004
U:min: -170.43
```

### price_range_by_side

```text
A:max: 199976.33
A:min: -170.42000000000002
B:max: 267.87
B:min: -569.77
```

## Interprétation

Les règles officielles sont documentées dans `docs/challenge_data_contract.md`. Les valeurs hors spread et les répétitions d'identifiant sont descriptives; elles ne sont pas corrigées automatiquement.
