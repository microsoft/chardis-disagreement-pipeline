# Table 1

Validation hypothesis F1 is primary. Heldout diagnostics expose the Yelp split discrepancy.

| dataset | method | judge | metric | row | expected | actual | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ghc-syn | chardis | o3 | hypothesis_f1 | 1 | 0.63 | 0.6310578975570762 | match |
| ghc-syn | chardis | o3 | hypothesis_f1 | 2 | 0.52 | 0.5218538630331006 | match |
| ghc-syn | chardis | o3 | hypothesis_f1 | 3 | 0.66 | 0.6636255672028063 | match |
| yelp | chardis | o3 | hypothesis_f1 | 1 | 0.74 | 0.7563483792828959 | mismatch |
| yelp | chardis | o3 | hypothesis_f1 | 2 | 0.63 | 0.4575996142754387 | mismatch |
| yelp | chardis | o3 | hypothesis_f1 | 3 | 0.54 | 0.5501467082747018 | mismatch |

