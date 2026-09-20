# Contrat sémantique des données

La source officielle est la [page ChallengeData du challenge 146](https://challengedata.ens.fr/challenges/146). Cette note la paraphrase; les constats locaux sont séparés dans le rapport de profilage.

Chaque `obs_id` désigne une séquence de 100 événements tirée d'une action et d'une journée aléatoires. `venue` identifie une place par un entier. `action` vaut `A` (ajout), `D` (suppression) ou `U` (modification). `order_id` est un identifiant obfusqué qui suit un ordre dans la séquence; le premier ordre référencé reçoit 0. `side` vaut `A` pour ask et `B` pour bid. `price` est le prix de l'ordre affecté; `bid` et `ask` sont les meilleurs prix consolidés; `bid_size` et `ask_size` sont leurs volumes. `flux` mesure la variation de volume due à l'événement. `trade` distingue une suppression ou modification liée à une transaction d'une annulation. La cible `eqt_code_cat` est une action anonymisée entière de 0 à 23.

La normalisation officielle est :

$$price'_t = price_t - bid_0$$
$$bid'_t = bid_t - bid_0$$
$$ask'_t = ask_t - bid_0$$

où `bid_0` est le meilleur bid du premier événement. Le train provient d'une période antérieure au test.

## Référence de benchmark

La page décrit une représentation `(100, 30)`, des embeddings pour `venue`, `action` et `trade`, des variables continues (`bid`, `ask`, `price`, tailles et `flux`), un GRU bidirectionnel, 24 logits et CrossEntropy. C'est une référence, pas l'architecture retenue ici; elle omet `side` et `order_id`. Sa mention de `log(flux)` est ambiguë puisque des flux négatifs sont observés. Aucune transformation logarithmique de `flux` ne sera appliquée avant clarification.

## Volumétrie publiée et locale

Les fichiers locaux donnent `160800 + 81600 = 242400` observations. La page évoque 20 observations par action et par jour, 24 actions et environ 505 jours, tout en mentionnant aussi 504 jours: cette incohérence rédactionnelle n'est pas résolue ici. Elle ne permet pas d'inférer des dates ni un ordre chronologique des identifiants.
