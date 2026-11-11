# Replication comparison summary

This is partial validation, heldout and corpus result replication. Convention values are counted separately from measured matches.

Processed 26 runs and 1082 hypotheses.
Checked 8881 stored metric values with 38 mismatches.

## Paper comparisons

| kind | status | count |
| --- | --- | --- |
| numeric | match | 129 |
| numeric | blocked | 32 |
| numeric | unsupported | 26 |
| numeric | mismatch | 18 |
| numeric | convention | 2 |
| hypothesis list | match | 175 |
| hypothesis list | blocked | 2 |

## Mismatches

Table 3 dices chardis o3 macro_f1 reports 0.58 and recomputes to 0.3260467980295566.

Table 3 dices hypothesaes o3 ci_half_width reports 0.06 and recomputes to 0.0653007800823257.

Table 3 ghc-real hypogenic o3 macro_f1 reports 0.48 and recomputes to 0.41910043049436063.

Table 3 md chardis o3 ci_half_width reports 0.02 and recomputes to 0.026229563626316343.

Table 3 md hypothesaes o3 ci_half_width reports 0.02 and recomputes to 0.02726628162227665.

Table 3 md hypogenic o3 ci_half_width reports 0.02 and recomputes to 0.027361660921806974.

Table 3 pref chardis o3 breadth reports 20 and recomputes to 9.

Table 3 pref hypothesaes o3 macro_f1 reports 0.32 and recomputes to 0.4411214953271028.

Table 5 ghc-real chardis o3 annotators reports 16 and recomputes to 18.

Table 6 dices chardis o3 corpus_rows reports 990 and recomputes to 991.

Table 1 yelp chardis o3 hypothesis_f1 reports 0.74 and recomputes to 0.7563483792828959.

Table 1 yelp chardis o3 hypothesis_f1 reports 0.63 and recomputes to 0.4575996142754387.

Table 1 yelp chardis o3 hypothesis_f1 reports 0.54 and recomputes to 0.5501467082747018.

Table 2 yelp chardis o3 macro_f1 reports 0.79 and recomputes to 0.763688086391703.

Table 2 yelp hypothesaes o3 macro_f1 reports 0.56 and recomputes to 0.5275318003594944.

Table 2 ghc-syn oracle o3 ci_half_width reports 0.02 and recomputes to 0.03436488069650884.

Table 2 ghc-syn hypothesaes o3 ci_half_width reports 0.02 and recomputes to 0.025804156142886336.

Table 2 ghc-syn hypothesaes deepseek-r1 macro_f1 reports 0.49 and recomputes to 0.5134856167715852.

## Limits

Missing ModernBERT predictions, manual semantic recall labels, DeepSeek oracle provenance, and the random baseline protocol block complete Tables 1 through 3 coverage. Transfer, additional discovery experiments, manual correctness and cost records remain unavailable.

See comparisons.csv for every registered numeric or list comparison and coverage.json for all tables and figures. A match for a list uses the explicitly documented editorial identity mapping where the paper shortened a hypothesis.

The process exits with code 1. Code 1 flags measured mismatches. Code 2 flags execution errors or unresolved required mappings when there are no measured mismatches.
