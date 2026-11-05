You will be provided two sets of rationales written by two annotators (e.g., A1 and A2) for labeling the same set of texts based on shared annotation guidelines.
 
Your task is to produce an exhaustive list of text-characteristic explanations describing what kinds of texts are most likely to produce disagreement between the annotators, based on the provided rationales.
 
Critical constraints:
 - Every explanation MUST be stated as a property of the text itself (observable textual features, cues, ambiguity patterns, framing, pragmatics, implied meaning, context dependence).
 - DO NOT mention annotators, “A1/A2,” “one annotator,” “the other annotator,” or any difference in annotator behavior. Do not use comparative or conditional phrasing like “while the other,” “one tends to,” or “X gets labeled by…”.
 - Each explanation must be a short, standalone sentence describing a disagreement-triggering text characteristic.

Output format: each explanation must start with "-" and be enclosed in quotes. No headers.
 
Method you must follow (internal reasoning, but outputs must obey constraints above):
 1. Atomize each annotator’s rationales into a list of atomic criteria (one criterion per line).
 2. Compare every atomic criterion from Annotator 1 against every atomic criterion from Annotator 2 and identify:
   - a) Contradictions: criteria that directly oppose each other.
   - b) Unshared criteria: criteria that appear in one annotator’s rationales but are not present anywhere in the other annotator’s rationales (i.e., the other set is neutral/does not invoke that cue).
 3. Convert each contradiction or unshared criterion into a text-characteristic disagreement trigger without referencing annotators. The explanation should describe the kind of text where that criterion matters

Coverage requirements:
 - Be exhaustive and include all possible fine-grained text characteristics that can trigger disagreements, as supported by the rationales.
 - Prefer concrete, checkable wording over abstract wording.