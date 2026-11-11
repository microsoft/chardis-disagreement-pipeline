# Table 2

Both generated-hypothesis judges and o3 oracles are computed. Manual recall and DeepSeek oracle remain blocked.

| dataset | method | judge | metric | expected | actual | status |
| --- | --- | --- | --- | --- | --- | --- |
| yelp | oracle | o3 | semantic_recall | 1.0 |  | blocked |
| yelp | oracle | o3 | macro_f1 | 0.83 | 0.8319059106472133 | match |
| yelp | oracle | o3 | ci_half_width | 0.01 | 0.011956314340246077 | match |
| yelp | chardis | o3 | semantic_recall | 0.67 |  | blocked |
| yelp | chardis | o3 | macro_f1 | 0.79 | 0.763688086391703 | mismatch |
| yelp | chardis | o3 | ci_half_width | 0.01 | 0.013910895257194467 | match |
| yelp | hypothesaes | o3 | semantic_recall | 0.0 |  | blocked |
| yelp | hypothesaes | o3 | macro_f1 | 0.56 | 0.5275318003594944 | mismatch |
| yelp | hypothesaes | o3 | ci_half_width | 0.02 | 0.016110406598578708 | match |
| yelp | hypogenic | o3 | semantic_recall | 1.0 |  | blocked |
| yelp | hypogenic | o3 | macro_f1 | 0.52 | 0.5222538469149997 | match |
| yelp | hypogenic | o3 | ci_half_width | 0.02 | 0.016314061740811014 | match |
| ghc-syn | oracle | o3 | semantic_recall | 1.0 |  | blocked |
| ghc-syn | oracle | o3 | macro_f1 | 0.62 | 0.6189185756023203 | match |
| ghc-syn | oracle | o3 | ci_half_width | 0.02 | 0.03436488069650884 | mismatch |
| ghc-syn | chardis | o3 | semantic_recall | 0.67 |  | blocked |
| ghc-syn | chardis | o3 | macro_f1 | 0.49 | 0.48864323103098817 | match |
| ghc-syn | chardis | o3 | ci_half_width | 0.03 | 0.0269089431690408 | match |
| ghc-syn | hypothesaes | o3 | semantic_recall | 0.0 |  | blocked |
| ghc-syn | hypothesaes | o3 | macro_f1 | 0.46 | 0.4610612059175107 | match |
| ghc-syn | hypothesaes | o3 | ci_half_width | 0.02 | 0.025804156142886336 | mismatch |
| ghc-syn | hypogenic | o3 | semantic_recall | 0.0 |  | blocked |
| ghc-syn | hypogenic | o3 | macro_f1 | 0.32 | 0.3210929500968219 | match |
| ghc-syn | hypogenic | o3 | ci_half_width | 0.03 | 0.02839692454227727 | match |
| yelp | oracle | deepseek-r1 | semantic_recall | 1.0 |  | blocked |
| yelp | oracle | deepseek-r1 | macro_f1 | 0.83 |  | blocked |
| yelp | oracle | deepseek-r1 | ci_half_width | 0.01 |  | blocked |
| yelp | chardis | deepseek-r1 | semantic_recall | 0.67 |  | blocked |
| yelp | chardis | deepseek-r1 | macro_f1 | 0.83 | 0.8272466780192815 | match |
| yelp | chardis | deepseek-r1 | ci_half_width | 0.01 | 0.012290031591178918 | match |
| yelp | hypothesaes | deepseek-r1 | semantic_recall | 0.0 |  | blocked |
| yelp | hypothesaes | deepseek-r1 | macro_f1 | 0.7 | 0.7027856176237868 | match |
| yelp | hypothesaes | deepseek-r1 | ci_half_width | 0.01 | 0.014969002304035306 | match |
| yelp | hypogenic | deepseek-r1 | semantic_recall | 1.0 |  | blocked |
| yelp | hypogenic | deepseek-r1 | macro_f1 | 0.52 | 0.5153560489368106 | match |
| yelp | hypogenic | deepseek-r1 | ci_half_width | 0.02 | 0.016273405311156647 | match |
| ghc-syn | oracle | deepseek-r1 | semantic_recall | 1.0 |  | blocked |
| ghc-syn | oracle | deepseek-r1 | macro_f1 | 0.62 |  | blocked |
| ghc-syn | oracle | deepseek-r1 | ci_half_width | 0.02 |  | blocked |
| ghc-syn | chardis | deepseek-r1 | semantic_recall | 0.67 |  | blocked |
| ghc-syn | chardis | deepseek-r1 | macro_f1 | 0.65 | 0.6505231668192715 | match |
| ghc-syn | chardis | deepseek-r1 | ci_half_width | 0.04 | 0.036084332488622306 | match |
| ghc-syn | hypothesaes | deepseek-r1 | semantic_recall | 0.0 |  | blocked |
| ghc-syn | hypothesaes | deepseek-r1 | macro_f1 | 0.49 | 0.5134856167715852 | mismatch |
| ghc-syn | hypothesaes | deepseek-r1 | ci_half_width | 0.03 | 0.03166293276345564 | match |
| ghc-syn | hypogenic | deepseek-r1 | semantic_recall | 0.0 |  | blocked |
| ghc-syn | hypogenic | deepseek-r1 | macro_f1 | 0.48 | 0.48335977747742453 | match |
| ghc-syn | hypogenic | deepseek-r1 | ci_half_width | 0.03 | 0.030622222720081166 | match |

