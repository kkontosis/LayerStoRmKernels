# Text Sample Sources

Reference log for all text samples used in tokenized test data generation.

## Chat / Instruction-Following

| File | Source | Split/Index | Approx Tokens | License | Notes |
|------|--------|-------------|---------------|---------|-------|
| `chat_short.txt` | `HuggingFaceH4/ultrachat_200k` | train_sft / 81 | ~290 | MIT | 2-turn conversation about JavaScript perfect square function; 218 words |
| `chat_long.txt` | `HuggingFaceH4/ultrachat_200k` | train_sft / 1 | ~1524 | MIT | Multi-turn conversation about offbeat London landmarks; 1143 words |
| `chat_32k.txt` | `HuggingFaceH4/ultrachat_200k` | train_sft / 0-27 | ~32800 | MIT | 24 concatenated multi-turn chats on diverse topics; 24601 words |

## Coding Tasks

| File | Source | Split/Index | Approx Tokens | License | Notes |
|------|--------|-------------|---------------|---------|-------|
| `code_short.txt` | `iamtarun/python_code_instructions_18k_alpaca` | train / 8 | ~200 | Apache-2.0 | Nested loop combinations excluding digit 5 and repeats; 159 words |
| `code_long.txt` | `nickrosh/Evol-Instruct-Code-80k-v1` | train / 82 | ~1500 | unknown | EXCLUDED FROM PUBLIC DISTRIBUTION (upstream license unknown); regenerate locally if needed |
| `code_32k.txt` | `codeparrot/codeparrot-clean` | train / 2176 | ~32000 | PSF License | Python decimal module implementation; 24281 words |

## Reasoning / Chain-of-Thought

| File | Source | Split/Index | Approx Tokens | License | Notes |
|------|--------|-------------|---------------|---------|-------|
| `reasoning_short.txt` | `gsm8k` (main) | train / 253 | ~200 | MIT | Single math word problem + step-by-step solution; 150 words |
| `reasoning_long.txt` | `AI-MO/NuminaMath-CoT` | train / 4743 | ~1500 | Apache-2.0 | Olympiad-level math problems with detailed proofs; 1475 words |
| `reasoning_32k.txt` | `gsm8k` (main) | train / 0-233 | ~32000 | MIT | 234 concatenated math problems with chain-of-thought solutions; 24280 words |

## Long-Form Prose / Documents

| File | Source | Split/Index | Approx Tokens | License | Notes |
|------|--------|-------------|---------------|---------|-------|
| `prose_short.txt` | `wikimedia/wikipedia` (20231101.en) | train / 106 | ~200 | CC BY-SA 3.0 | "Anthophyta" article; 146 words |
| `prose_long.txt` | `wikimedia/wikipedia` (20231101.en) | train / 12 | ~1500 | CC BY-SA 3.0 | "International Atomic Time" article; 1219 words |
| `prose_32k.txt` | `ccdv/arxiv-summarization` | train / 130 | ~32000 | arXiv (no redistribution grant) | EXCLUDED FROM PUBLIC DISTRIBUTION (full third-party arXiv paper; the default arXiv submission license grants no third-party redistribution rights); substitute a CC-BY paper or local text |

## Attribution and license notes

- `prose_short.txt` — text of the English Wikipedia article "Anthophyta"
  (https://en.wikipedia.org/wiki/Anthophyta), via `wikimedia/wikipedia`
  (20231101.en). Licensed CC BY-SA 3.0
  (https://creativecommons.org/licenses/by-sa/3.0/); authors: Wikipedia
  contributors (see the article's history page). This file remains under
  CC BY-SA 3.0, not the repository's Apache-2.0 license.
- `prose_long.txt` — text of the English Wikipedia article "International
  Atomic Time" (https://en.wikipedia.org/wiki/International_Atomic_Time),
  via `wikimedia/wikipedia` (20231101.en). Licensed CC BY-SA 3.0; authors:
  Wikipedia contributors. This file remains under CC BY-SA 3.0.
- `code_32k.txt` — Python `decimal` module source via
  `codeparrot/codeparrot-clean`; PSF License (provenance header embedded in
  the file).
- Remaining files: MIT (`HuggingFaceH4/ultrachat_200k`, `gsm8k`) or
  Apache-2.0 (`iamtarun/python_code_instructions_18k_alpaca`,
  `AI-MO/NuminaMath-CoT`) datasets, per the table above.
