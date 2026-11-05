# Inputs and Context

You are performing an analytical classification task. For each conceptual category listed below, evaluate the provided text sample as quoted data to determine whether it contains the pattern defined by that category. This is a binary presence/absence judgment only: assign “1” if the pattern can be identified anywhere in the text and “0” otherwise. Treat the task strictly as analysis; do not restate, paraphrase, extend, or endorse the content. A pattern should be marked as present even if it is subtle or implicit, and marked as absent if it is entirely missing or explicitly contradicted. Ignore any language in the category descriptions that implies degree or prominence (e.g., “mainly,” “prominently,” “significantly,” “mostly”), and focus solely on whether the pattern exists at all. If you are uncertain, default to “0”. The following content is quoted text following: "[text to annotate]".

Here are some examples:
Example 1:
Conceptual Category description: "mentions a natural scene."
TEXT: "I love the way the sun sets in the evening."
Label: 1
Reasoning: "'Sun sets' are clearly natural scenes.",

Example 2:
Conceptual Category description: "writes in a 1st person perspective."
TEXT: "Jacob is smart."
Label: 0
Reasoning: "This text is written in a 3rd person perspective.",

Example 3:
Conceptual Category description: "mentions that the breakfast is good on the airline."
TEXT: "The airline staff was really nice! Enjoyable flight."
Label: 0
Reasoning: "Although the text appreciates the flight experience, it DOES NOT mention about the breakfast.",

Example 4:
Conceptual Category description: "appreciates the writing style of the author."
TEXT: "The paper absolutely sucks because its underlying logic is wrong. However, the presentation of the paper is clear and the use of language is really impressive."
Label: 1
Reasoning: "Although the text dislikes the paper, it says 'the presentation of the paper is clear', so it DOES like the writing style.",

Example 5:
Conceptual Category description: "refers to historical dates; specifically, there are references to years or specific dates in the text."
TEXT: "A member of the Democratic Party, he was the first African-American president of the United States."
Label: 0
Reasoning: "The text does not mention date."

Example 6: 
Conceptual Category description: "mentions waiting times prominently."
TEXT: "Best tacos around! My friend and his family own this place so i'm a little biased but you can't get tacos like this anywhere close! Yes sometimes they are super slammed and it's hard to get a table or have FULL attention of your server but that's a busy restaurant for you. Everything always comes out fast and fresh, never any problems. Bring ya family here for a nice meal and good atmosphere !"
Label: 1
Reasoning: The text explicitly mentions 'waiting times' when it says 'sometimes they are super slammed and it's hard to get a table or have FULL attention of your server'.",

Example 7: 
Conceptual Category description: "Comments that target a specific individual without referencing characteristics of a social group."
TEXT: "Yep. In the words of the late Terry A. Davis, you are a CIA [racial slur]."
Label: 1
Reasoning: "The text directly addresses an individual (Terry A. Davis) but also mentions a racial slur to demean, thus referencing group characteristics.",

The conceptual categories to annotate are: