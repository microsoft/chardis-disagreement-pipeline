# Replication guideline index

These seven dataset guidelines were copied without changing their bytes. The original source files remain in place.

| Guideline | Task |
| --- | --- |
| [DICES](dices/prompt.md) | Conversation safety judgments |
| [GHC real](ghc-real/prompt.md) | Hate speech judgments using human corpus guidelines |
| [GHC Synthetic](ghc-syn/prompt.md) | Synthetic hate speech labeling criteria |
| [MD Agreement](md/prompt.md) | Offensive language judgments |
| [MHSC](mhsc/prompt.md) | Hate speech judgments |
| [MultiPref](pref/prompt.md) | Preference judgments between response options |
| [Yelp Synthetic](yelp/prompt.md) | Food, staff, and time categories |

These documents are labeling guidelines. They do not contain the full generated hypothesis judge prompts or cached model responses.

The [copied configurations](../replication_configs/INDEX.md) reference these paths. The pipeline's separate discovery and annotation templates remain under [code-repo-final/prompts](../code-repo-final/prompts/).

Prompt text does not establish how historical disagreement targets were constructed. In particular, MHSC has a guideline and configs but no result family in the authorized source tree.

The [conversion report](../outputs/config_conversion.json) records checksums for all seven copies. See [REPLICATION.md](../REPLICATION.md) for offline replication coverage and limitations.
