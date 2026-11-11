# Table 16

Baseline validation hypothesis lists

| dataset | method | generator | judge | split | rank | hypothesis_id | hypothesis | recomputed_f1 | p_value_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ghc-real | hypothesaes | o3 | o3 | validation | 1 | h01 | mentions explicit US political-orientation terms (e.g., “left”, “leftist(s)”, “right”, “right-wing”, “conservative”) or names the news networks “CNN” or “MSNBC” | 0.5517115319422676 | 1.5195616660439113e-14 |
| ghc-real | hypothesaes | o3 | o3 | validation | 2 | h02 | Contains words referring to left-wing ideologies (e.g., “socialist”, “communist”, “marxist”, “collectivist”) | 0.5239930185642723 | 1.1079439788366919e-14 |
| ghc-real | hypothesaes | o3 | o3 | validation | 3 | h03 | mentions Islam/Muslims (e.g., words like 'Islam', 'Islamic', 'muslim') | 0.5111449181558823 | 1.09879180127736e-14 |
| ghc-real | hypothesaes | o3 | o3 | validation | 4 | h04 | contains a low-intelligence insult such as “stupid”, “idiot”, “moron”, or “dumb” | 0.5055324568880099 | 4.492093797797206e-10 |
| ghc-real | hypothesaes | o3 | o3 | validation | 5 | h05 | Contains a word that explicitly refers to women or female identity (e.g., 'woman', 'women', 'daughters', 'burka', etc.) | 0.5052377496321115 | 1.721697558643441e-05 |
| ghc-real | hypothesaes | o3 | o3 | validation | 6 | h06 | contains explicit sexual or pornographic terms (e.g., 'porn', 'sex', 'pedo', 'naked', '#NSFW') | 0.4985333317104639 | 4.316637961633253e-09 |
| ghc-real | hypothesaes | o3 | o3 | validation | 7 | h07 | mentions a U.S. President (surname or the words 'President'/'POTUS') | 0.4941172629218864 | 0.034851215240254606 |
| ghc-real | hypothesaes | o3 | o3 | validation | 8 | h08 | mentions the word “white” (or “whites”) when referring to people | 0.49356612511131726 | 2.962339395518553e-12 |
| ghc-real | hypothesaes | o3 | o3 | validation | 9 | h09 | contains terms explicitly referring to women/feminism (e.g., 'woman', 'women', 'feminist') | 0.48979494706622584 | 0.00010319526387529433 |
| ghc-real | hypothesaes | o3 | o3 | validation | 10 | h10 | contains a hostile imperative command (e.g., 'fuck off', 'shut...', 'run along', 'go fuck', 'fight me', 'drag him', 'gab away') | 0.4876808551952696 | 2.2180448933781627e-05 |
| ghc-real | hypothesaes | o3 | o3 | validation | 11 | h11 | contains the word 'jew' (case-insensitive) | 0.48237941793258654 | 2.313029071230733e-10 |
| ghc-real | hypothesaes | o3 | o3 | validation | 12 | h12 | contains a reference to children (e.g., the words 'child', 'children', or 'kids') | 0.4806594741112674 | 0.00011180332485745639 |
| ghc-real | hypothesaes | o3 | o3 | validation | 13 | h13 | Contains language about falseness, featuring words like 'fake', 'hoax', 'debunk(ed)', or the word 'truth' | 0.47699464607677167 | 0.0006106872798806924 |
| ghc-real | hypothesaes | o3 | o3 | validation | 14 | h14 | contains an explicit quantitative statistic (e.g., specific numbers, percentages, counts, rankings, or words like “seven”) | 0.47357334745438817 | 0.025323568560566622 |
| ghc-real | hypothesaes | o3 | o3 | validation | 15 | h15 | mentions elections or voting (words such as vote/voting/election/re-election/Senate seat) | 0.4676614168881458 | 0.02324919474561668 |
| ghc-real | hypothesaes | o3 | o3 | validation | 16 | h16 | mentions the word stem “America” (e.g., America, American, Americans) | 0.4643230547155014 | 0.003654070558227008 |
| ghc-real | hypothesaes | o3 | o3 | validation | 17 | h17 | contains the word stem “liber-” (e.g., “liberal”, “libs”, “libtards”) | 0.46174118445922585 | 0.00020713588457997486 |
| ghc-real | hypothesaes | o3 | o3 | validation | 18 | h18 | contains the pronoun 'she' | 0.45646338500013617 | 0.01708831615327791 |
| ghc-real | hypothesaes | o3 | o3 | validation | 19 | h19 | contains the word “gay”, “fag”, or “faggot” | 0.4515098418769034 | 2.0154331948884196e-05 |
| ghc-real | hypothesaes | o3 | o3 | validation | 20 | h20 | message is written entirely in uppercase letters (all alphabetic characters are uppercase) | 0.4514778573590235 | 0.004292946114947289 |
| ghc-real | hypogenic | o3 | o3 | validation | 1 | h01 | Mild or borderline profanity/hate speech: Posts that include mild slurs, reclaimed slurs, or profanity used non-aggressively can cause disagreement about whether the text is offensive or acceptable. | 0.6710487204587879 | 5.87311983866357e-53 |
| ghc-real | hypogenic | o3 | o3 | validation | 2 | h02 | Borderline or subtle offensive language (mild profanity, indirect slurs) results in disagreement about whether it crosses a toxicity threshold. | 0.6448895503942751 | 1.8666899411067663e-38 |
| ghc-real | hypogenic | o3 | o3 | validation | 3 | h03 | Borderline or context-dependent offensive language (e.g., mild profanity, reclaimed slurs) often leads to uncertainty about whether the content violates guidelines, driving disagreement. | 0.6350247862288033 | 7.98152589797257e-34 |
| ghc-real | hypogenic | o3 | o3 | validation | 4 | h04 | Texts containing mild, borderline, or context-dependent profanity/offensiveness (e.g., reclaimed slurs, joking insults) lead to higher disagreement because annotators differ on whether they cross the threshold for toxicity. | 0.6332937832370189 | 1.8253740846150064e-33 |
| ghc-real | hypogenic | o3 | o3 | validation | 5 | h05 | Texts that contain mild profanity or borderline offensive terms spark disagreement over whether they are acceptable, hateful, or merely emphatic. | 0.6261367511854461 | 9.936551433763053e-32 |
| ghc-real | hypogenic | o3 | o3 | validation | 6 | h06 | Texts addressing controversial or polarizing topics (politics, religion, identity issues) tend to show higher disagreement because annotators may project their own biases. | 0.5876103354196758 | 2.694336612194354e-44 |
| ghc-real | hypogenic | o3 | o3 | validation | 7 | h07 | Texts discussing morally charged or sensitive topics (e.g., politics, religion, social justice) trigger higher disagreement because personal beliefs influence interpretation. | 0.5796682140194458 | 8.617840164331469e-43 |
| ghc-real | hypogenic | o3 | o3 | validation | 8 | h08 | Figurative or metaphorical language: When authors use metaphors, idioms, or other figurative devices, the literal meaning diverges from the intended meaning, increasing chances of differing interpretations. | 0.552381745502998 | 5.814543053803462e-08 |
| ghc-real | hypogenic | o3 | o3 | validation | 9 | h09 | Texts referencing cultural, political, or regional issues unfamiliar to all annotators create disagreement through uneven background knowledge. | 0.544398997677795 | 4.5776023224394865e-07 |
| ghc-real | hypogenic | o3 | o3 | validation | 10 | h10 | Use of figurative or metaphorical language (e.g., idioms, creative metaphors) raises disagreement by obscuring literal meaning. | 0.5416369927181557 | 5.908626079172288e-05 |
| ghc-real | hypogenic | o3 | o3 | validation | 11 | h11 | Texts containing sarcasm without clear markers (e.g., no emojis, no quotation marks) lead to more disagreement because sarcasm requires contextual inference that varies by annotator. | 0.540859273367287 | 4.7306608159049185e-08 |
| ghc-real | hypogenic | o3 | o3 | validation | 12 | h12 | Texts that contain sarcasm, irony, or humor cues without explicit markers (e.g., emojis, quotation marks) are more likely to yield annotator disagreement. | 0.540372421728354 | 2.5081588547265595e-06 |
| ghc-real | hypogenic | o3 | o3 | validation | 13 | h13 | Long, multi-clause sentences with complex syntax increase disagreement because different annotators may focus on different parts of the text. | 0.5394666074435448 | 3.820279600908874e-08 |
| ghc-real | hypogenic | o3 | o3 | validation | 14 | h14 | Texts that contain ambiguous or metaphoric language (e.g., figurative expressions, sarcasm, irony) are more likely to produce annotator disagreement because annotators may interpret the intended meaning differently. | 0.5322451442187646 | 6.71306956983228e-05 |
| ghc-real | hypogenic | o3 | o3 | validation | 15 | h15 | Texts that contain sarcasm, irony, or humor, making the author’s true intent ambiguous, are more likely to trigger annotator disagreement. | 0.5320176681783824 | 0.0005272113285230436 |
| ghc-real | hypogenic | o3 | o3 | validation | 16 | h16 | Use of domain-specific jargon, slang, or newly coined terms increases disagreement because annotators may not share the same background knowledge. | 0.5303516195753241 | 0.001838665211294659 |
| ghc-real | hypogenic | o3 | o3 | validation | 17 | h17 | Presence of negations or double negatives can confuse meaning and increase annotator disagreement. | 0.5291576617526029 | 0.00908174889471261 |
| ghc-real | hypogenic | o3 | o3 | validation | 18 | h18 | Negation or double negatives: Sentences containing negations (e.g., “I don’t dislike it”) or double negatives can be hard to parse, causing differing judgments about the actual stance or sentiment. | 0.5251888634241576 | 0.023114238432800374 |
| ghc-real | hypogenic | o3 | o3 | validation | 19 | h19 | Texts that use negations or double negatives (“not unhappy,” “didn’t dislike”) cause disagreement since annotators may differ in how they parse the logical sentiment. | 0.5242833395043431 | 0.028060226559783144 |
| ghc-real | hypogenic | o3 | o3 | validation | 20 | h20 | Presence of sarcasm or irony: Texts that contain sarcastic remarks, irony, or subtle humor (e.g., using praise to mean criticism) are more likely to confuse annotators and lead to disagreement. | 0.5231235734817241 | 6.501490082796909e-05 |

F1 values in these list exports are recomputed validation diagnostics. The appendix list tables do not print F1 values.

