# Table 19

Baseline validation hypothesis lists

| dataset | method | generator | judge | split | rank | hypothesis_id | hypothesis | recomputed_f1 | p_value_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md | hypothesaes | o3 | o3 | validation | 1 | h01 | mentions COVID-19 or coronavirus | 0.5387186266790445 | 1.8565892651522843e-06 |
| md | hypothesaes | o3 | o3 | validation | 2 | h02 | mentions the term “COVID-19” | 0.5221905301178548 | 1.2473354427494318e-06 |
| md | hypothesaes | o3 | o3 | validation | 3 | h03 | contains the word 'Trump' | 0.4920410783055199 | 0.0049195697587620804 |
| md | hypothesaes | o3 | o3 | validation | 4 | h04 | contains one of the insult words {moron, stfu, evil, arseholes, crap, shit, loser, joke, lying, trash, stupid} | 0.48966004909808947 | 6.274524989167746e-05 |
| md | hypothesaes | o3 | o3 | validation | 5 | h05 | accuses someone of lying (uses words like lie, liar, lying, lies, dishonest) | 0.4510147503081502 | 2.9944996064349602e-05 |
| md | hypothesaes | o3 | o3 | validation | 6 | h06 | explicitly names a U.S. political party (e.g., the words ‘Democrat(s)’, ‘Dems’, ‘Republican(s)’, or ‘GOP’ appear) | 0.43406628857531815 | 0.009490471424224193 |
| md | hypothesaes | o3 | o3 | validation | 7 | h07 | contains a standalone word that is a variant of “lie” (e.g., ‘lie’, ‘lies’, ‘lying’, ‘lied’, or ‘liar’) | 0.43401899876461797 | 0.0031722797387895327 |
| md | hypothesaes | o3 | o3 | validation | 8 | h08 | mentions Biden (the word “Biden” appears, including inside hashtags like #BidenIsWinnig) | 0.42439049552098024 | 0.016570626315258328 |
| md | hypothesaes | o3 | o3 | validation | 9 | h09 | mentions the surname 'Biden' | 0.42439049552098024 | 0.016570626315258328 |
| md | hypothesaes | o3 | o3 | validation | 10 | h10 | contains the substring “shit” (e.g., ‘shit’, ‘shitty’) | 0.4194852684098602 | 0.036070298164365876 |
| md | hypothesaes | o3 | o3 | validation | 11 | h11 | Contains the word “Biden” but does not contain the word “Vote” | 0.41425164161826644 | 0.028694582229475133 |
| md | hypothesaes | o3 | o3 | validation | 12 | h12 | mentions a protest or public demonstration | 0.4129263493253374 | 6.933922191834099e-06 |
| md | hypothesaes | o3 | o3 | validation | 13 | h13 | contains text enclosed in parentheses ( … ) | 0.40246869591774415 | 0.03174443049085837 |
| md | hypothesaes | o3 | o3 | validation | 14 | h14 | mentions George Floyd (the name or hashtag) | 0.3994438309852257 | 0.001336040593677262 |
| md | hypothesaes | o3 | o3 | validation | 15 | h15 | mentions China/Chinese | 0.39940754737574524 | 0.03097804263943427 |
| md | hypothesaes | o3 | o3 | validation | 16 | h16 | includes the hashtag #BlackLivesMatter (any capitalization) | 0.3961681181141859 | 0.00011048530163277888 |
| md | hypothesaes | o3 | o3 | validation | 17 | h17 | contains the hashtag #BlackLivesMatter | 0.3961681181141859 | 0.00011048530163277888 |
| md | hypothesaes | o3 | o3 | validation | 18 | h18 | contains the hashtag #BlackLivesMatter (case-insensitive) | 0.3961681181141859 | 0.00011048530163277888 |
| md | hypothesaes | o3 | o3 | validation | 19 | h19 | mentions the name of a city (e.g., New York, LA, Chicago, Naples, Brooklyn, Milano) | 0.394319282225273 | 0.008767702497254122 |
| md | hypothesaes | o3 | o3 | validation | 20 | h20 | includes the hashtag “#BlackLivesMatter” without also including the hashtag “#BLM” | 0.3937977793957043 | 6.696093942224566e-05 |
| md | hypogenic | o3 | o3 | validation | 1 | h01 | Borderline policy violations: Content that is borderline abusive, hateful, or disinformative (just shy of or just over policy boundaries) leads to differing judgments about severity or compliance. | 0.6453540491077787 | 4.0908684259443335e-26 |
| md | hypogenic | o3 | o3 | validation | 2 | h02 | Statements that can be interpreted as either factual or opinionated (e.g., “It’s the worst movie this year”) cause annotators to diverge on labeling objective vs. subjective content. | 0.5829808244358707 | 1.4293867373407656e-09 |
| md | hypogenic | o3 | o3 | validation | 3 | h03 | Texts that include borderline or subtle slurs, insults, or coded language (e.g., “dog-whistles”) create uncertainty about whether they should be labeled as hateful/abusive, increasing disagreement. | 0.5660613238584025 | 2.1496505252295136e-10 |
| md | hypogenic | o3 | o3 | validation | 4 | h04 | Texts that include figurative language (metaphors, idioms, hyperbole) are more prone to disagreement because literal vs. figurative readings differ. | 0.5647365795633321 | 2.388240089928712e-06 |
| md | hypogenic | o3 | o3 | validation | 5 | h05 | Mentions of multiple entities, events, or viewpoints in the same passage increase ambiguity, driving disagreement over which target or stance should be labeled. | 0.5574427848089386 | 2.265060746393522e-05 |
| md | hypogenic | o3 | o3 | validation | 6 | h06 | Mixed references to multiple entities or targets (e.g., multiple people or groups discussed in one sentence) make it harder to decide which target a label applies to, fostering disagreement. | 0.5535575399683134 | 9.03493900358989e-05 |
| md | hypogenic | o3 | o3 | validation | 7 | h07 | Figurative or idiomatic language: Sentences using metaphors, idioms, or slang often require cultural or contextual knowledge, making annotator interpretations diverge. | 0.5521248676814268 | 0.00011688026674859701 |
| md | hypogenic | o3 | o3 | validation | 8 | h08 | Texts that contain negations, sarcasm markers, or rhetorical questions (e.g., “You don’t actually believe that, do you?”) can obscure the underlying stance, leading to labeling inconsistencies. | 0.5423660471484031 | 0.0016323097046059372 |
| md | hypogenic | o3 | o3 | validation | 9 | h09 | Texts that contain sarcasm, irony, or other figurative language are more likely to trigger annotator disagreement because the true intent can be interpreted in multiple ways. | 0.5289173977425596 | 2.0110762337457506e-05 |
| md | hypogenic | o3 | o3 | validation | 10 | h10 | Texts with ambiguous or shifting targets (unclear who “they,” “you,” or “we” refer to) often cause annotators to diverge in labeling. | 0.5272777954762664 | 0.041855948077173584 |
| md | hypogenic | o3 | o3 | validation | 11 | h11 | Offensive or profane expressions used non-literally (e.g., jokingly or affectionately) create ambiguity about intent and raise disagreement. | 0.5272042575443213 | 1.686646620983559e-07 |
| md | hypogenic | o3 | o3 | validation | 12 | h12 | Texts with negations or double negatives (e.g., “I don’t hate it”) lead to different sentiment or stance judgments and thus more disagreement. | 0.5230973093781165 | 0.03518640611429114 |
| md | hypogenic | o3 | o3 | validation | 13 | h13 | Texts with implicit hate speech, micro-aggressions, or subtle offenses (rather than explicit slurs) make it harder for annotators to agree on offense levels, raising disagreement likelihood. | 0.5143471997311799 | 1.94400702145133e-06 |
| md | hypogenic | o3 | o3 | validation | 14 | h14 | Texts containing sarcasm, irony, or humor cues (e.g., “yeah, right…”) tend to confuse annotators and increase disagreement rates. | 0.5136109459965216 | 0.0005081877800353862 |
| md | hypogenic | o3 | o3 | validation | 15 | h15 | Texts referencing specific cultural, historical, or political knowledge (e.g., niche memes, historical events) can cause disagreement if annotators differ in familiarity with the context. | 0.5085602295954855 | 0.003969995672239411 |
| md | hypogenic | o3 | o3 | validation | 16 | h16 | Humor and irony: Jokes, memes, or ironic statements rely on implicit cues; without shared humor context, annotators often disagree on literal vs. intended meaning. | 0.49319133371449275 | 0.005497725565021549 |
| md | hypogenic | o3 | o3 | validation | 17 | h17 | Texts that express mixed or conflicting sentiments (e.g., praise and criticism in the same message) tend to confuse annotators, leading to higher disagreement. | 0.4892363389747957 | 0.01886107117351764 |
| md | hypogenic | o3 | o3 | validation | 18 | h18 | Sentences that mix conflicting cues (e.g., both positive and negative language) foster disagreement on overall polarity or intent. | 0.4890183517895589 | 0.025568798197078742 |
| md | hypogenic | o3 | o3 | validation | 19 | h19 | Texts discussing controversial or polarizing topics (e.g., politics, religion, social justice issues) provoke stronger subjective reactions and hence greater annotator disagreement. | 0.4828477221623868 | 0.000266652766939544 |
| md | hypogenic | o3 | o3 | validation | 20 | h20 | Texts containing sarcasm, irony, or other forms of indirect meaning cue higher disagreement because annotators differ on whether the intent is literal or not. | 0.47940512464280644 | 0.0002479146047102485 |

F1 values in these list exports are recomputed validation diagnostics. The appendix list tables do not print F1 values.

