# Table 3

Heldout F1 is recomputed from validation-selected features and thresholds. ModernBERT and random baselines remain blocked. Empty MultiPref HypoGeniC uses a declared convention.

| dataset | method | judge | metric | expected | actual | status |
| --- | --- | --- | --- | --- | --- | --- |
| dices | chardis | o3 | breadth | 16.0 | 16.0 | match |
| dices | chardis | o3 | macro_f1 | 0.58 | 0.3260467980295566 | mismatch |
| dices | chardis | o3 | ci_half_width | 0.06 | 0.06331284961828362 | match |
| dices | hypothesaes | o3 | breadth | 9.0 | 9.0 | match |
| dices | hypothesaes | o3 | macro_f1 | 0.65 | 0.6536392405063292 | match |
| dices | hypothesaes | o3 | ci_half_width | 0.06 | 0.0653007800823257 | mismatch |
| dices | hypogenic | o3 | breadth | 4.0 | 4.0 | match |
| dices | hypogenic | o3 | macro_f1 | 0.62 | 0.6210779037999834 | match |
| dices | hypogenic | o3 | ci_half_width | 0.07 | 0.07322938364118636 | match |
| ghc-real | chardis | o3 | breadth | 41.0 | 41.0 | match |
| ghc-real | chardis | o3 | macro_f1 | 0.71 | 0.7127112476105926 | match |
| ghc-real | chardis | o3 | ci_half_width | 0.02 | 0.021511443264616903 | match |
| ghc-real | hypothesaes | o3 | breadth | 27.0 | 27.0 | match |
| ghc-real | hypothesaes | o3 | macro_f1 | 0.62 | 0.6193441890491094 | match |
| ghc-real | hypothesaes | o3 | ci_half_width | 0.02 | 0.02291724329218181 | match |
| ghc-real | hypogenic | o3 | breadth | 33.0 | 33.0 | match |
| ghc-real | hypogenic | o3 | macro_f1 | 0.48 | 0.41910043049436063 | mismatch |
| ghc-real | hypogenic | o3 | ci_half_width | 0.02 | 0.021916098218558 | match |
| md | chardis | o3 | breadth | 32.0 | 32.0 | match |
| md | chardis | o3 | macro_f1 | 0.63 | 0.6273959681659906 | match |
| md | chardis | o3 | ci_half_width | 0.02 | 0.026229563626316343 | mismatch |
| md | hypothesaes | o3 | breadth | 25.0 | 25.0 | match |
| md | hypothesaes | o3 | macro_f1 | 0.5 | 0.5031837134269913 | match |
| md | hypothesaes | o3 | ci_half_width | 0.02 | 0.02726628162227665 | mismatch |
| md | hypogenic | o3 | breadth | 28.0 | 28.0 | match |
| md | hypogenic | o3 | macro_f1 | 0.57 | 0.5727821784085014 | match |
| md | hypogenic | o3 | ci_half_width | 0.02 | 0.027361660921806974 | mismatch |
| pref | chardis | o3 | breadth | 20.0 | 9.0 | mismatch |
| pref | chardis | o3 | macro_f1 | 0.51 | 0.5050167645297255 | match |
| pref | chardis | o3 | ci_half_width | 0.02 | 0.02192015630648203 | match |
| pref | hypothesaes | o3 | breadth | 7.0 | 7.0 | match |
| pref | hypothesaes | o3 | macro_f1 | 0.32 | 0.4411214953271028 | mismatch |
| pref | hypothesaes | o3 | ci_half_width | 0.01 | 0.00559721690241477 | match |
| pref | hypogenic | o3 | breadth | 0.0 | 0.0 | match |
| pref | hypogenic | o3 | macro_f1 | 0.0 | 0.0 | convention |
| pref | hypogenic | o3 | ci_half_width | 0.0 | 0.0 | convention |
| md | random | o3 | macro_f1 | 0.5 |  | blocked |
| md | modernbert | o3 | macro_f1 | 0.61 |  | blocked |
| md | modernbert | o3 | ci_half_width | 0.03 |  | blocked |
| pref | random | o3 | macro_f1 | 0.5 |  | blocked |
| pref | modernbert | o3 | macro_f1 | 0.6 |  | blocked |
| pref | modernbert | o3 | ci_half_width | 0.02 |  | blocked |
| dices | random | o3 | macro_f1 | 0.5 |  | blocked |
| dices | modernbert | o3 | macro_f1 | 0.69 |  | blocked |
| dices | modernbert | o3 | ci_half_width | 0.07 |  | blocked |
| ghc-real | random | o3 | macro_f1 | 0.5 |  | blocked |
| ghc-real | modernbert | o3 | macro_f1 | 0.7 |  | blocked |
| ghc-real | modernbert | o3 | ci_half_width | 0.02 |  | blocked |

