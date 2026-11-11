# Table 14

o3 lists supported; R1 source pairs missing

| dataset | method | generator | judge | split | rank | hypothesis_id | hypothesis | recomputed_f1 | p_value_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ghc-real | chardis | o3 | o3 | validation | 1 | h01 | "Texts that portray a protected group in a uniformly negative or contemptuous light without using explicit slurs or advocating violence." | 0.5994823561880817 | 2.585989171810037e-33 |
| ghc-real | chardis | o3 | o3 | validation | 2 | h02 | "Texts where hostility targets a collective entity that is not clearly tied to a protected characteristic, creating ambiguity about hate relevance." | 0.5972493834653854 | 1.3660396887735852e-19 |
| ghc-real | chardis | o3 | o3 | validation | 3 | h03 | "Harsh accusations of wrongdoing directed at corporations, governments, or other non-protected collectives using charged language." | 0.5919619573554226 | 3.537274274952573e-21 |
| ghc-real | hypothesaes | o3 | o3 | validation | 1 | h01 | mentions explicit US political-orientation terms (e.g., “left”, “leftist(s)”, “right”, “right-wing”, “conservative”) or names the news networks “CNN” or “MSNBC” | 0.5517115319422676 | 1.5195616660439113e-14 |
| ghc-real | hypothesaes | o3 | o3 | validation | 2 | h02 | Contains words referring to left-wing ideologies (e.g., “socialist”, “communist”, “marxist”, “collectivist”) | 0.5239930185642723 | 1.1079439788366919e-14 |
| ghc-real | hypothesaes | o3 | o3 | validation | 3 | h03 | mentions Islam/Muslims (e.g., words like 'Islam', 'Islamic', 'muslim') | 0.5111449181558823 | 1.09879180127736e-14 |

F1 values in these list exports are recomputed validation diagnostics. The appendix list tables do not print F1 values.

chardis with deepseek-r1 remains blocked. Required v13 CSV pair is missing

hypothesaes with deepseek-r1 remains blocked. Required v14 CSV pair is missing

