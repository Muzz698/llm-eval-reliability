# Pre-registration: How reliable is an MMLU score? A generalizability-theory audit of small open LLMs

Author: Muzamil (independent researcher). Written on 19 November 2026, after a pilot and **before** collecting the main data. The design file `configs/design.yaml` and the analysis code in `src/evalrel/` are frozen in the same commit.

## 1. Question
Benchmark scores for LLMs are usually reported from a single run with one prompt template and one option order. Psychometrics treats a score as an observation that contains measurement error, and asks how large that error is. This study estimates how much of an MMLU score's variation comes from arbitrary choices that should not matter: the option order and the prompt layout. It then asks how large a difference between two models must be before it exceeds that error.

Prior work already shows that option order and prompt format move LLM scores (Zheng et al., 2024; Pezeshkpour & Hruschka, 2024; Sclar et al., 2024; Alzahrani et al., 2024). This study does not claim that finding as new. Its contribution is to express it in psychometric terms: generalizability-theory variance components, the standard error of measurement, and the minimal detectable difference.

## 2. Pilot (data discarded)
A pilot with 40 items and all four models was used only to check the pipeline and the time cost. It showed that the original MMLU framing ("Answer:" with no instruction) made models write sentences instead of letters, so up to 100% of answers could not be scored. All three templates were therefore given the same final instruction, after which 95–100% of answers could be parsed. Pilot data are not used in any analysis. The 40 pilot items are part of the 570 main items and will be queried again.

## 3. Design
- **Items:** 570 MMLU test items, 10 drawn at random from each of the 57 subjects (seed 0; `data/mmlu_570.jsonl`).
- **Models:** llama3.2:3b, qwen2.5:3b, qwen2.5:7b and qwen3:4b-instruct-2507 (q4_K_M), run locally with Ollama 0.40.16, in chat mode, with thinking disabled.
- **Facet arm (greedy decoding, temperature 0):** fully crossed items × 3 templates × 4 option orders.
  - Templates: `plain`, `mmlu` (the original paper's subject framing) and `paren` (options shown as "(A)"). All three end with "Answer with the letter of the correct option only."
  - Orders: 4 cyclic shifts of the options, so the correct answer appears once in each position A–D.
- **Retest arm:** the `plain` template with the original option order, at temperature 0.7, seeds 1–3.
- **Scoring:** the first clearly stated letter A–D (`parse_letter`). An answer that cannot be parsed counts as incorrect.

## 4. Hypotheses and planned tests
- **H1 (condition effect):** For at least one model, accuracy differs across the 12 template × order conditions.
  - Test: Cochran's Q per model, Holm-corrected across the 4 models, α = .05.
- **H2 (format error persists at scale):** For every model, the standard error of measurement due to format, extrapolated to the full 14,042-item MMLU test set, is larger than the binomial item-sampling standard error at that size.
  - SEM_format = sqrt(σ²_t + σ²_r + σ²_tr + (σ²_it + σ²_ir + σ²_itr,e) / n).
  - Item-sampling SE = sqrt(p(1−p) / n).
- **H3 (position bias):** For at least one model, the distribution of chosen letters differs from uniform.
  - Test: χ² goodness of fit, Holm-corrected, α = .05. The design guarantees that correct answers are uniform across positions.
- **H4 (ranking instability):** At least one model pair whose overall difference has a 95% bootstrap CI that includes 0 changes rank across the 12 conditions.
  - Paired item bootstrap, 2,000 resamples.
- **Primary descriptive outcome:** the minimal detectable difference, MDD95 = 1.96 × √2 × SEM_format, for each model at the sample size (570) and at full MMLU size.

## 5. Variance components
A random-effects G-study (Brennan, 2001) for each model, fitted to 0/1 correctness in the fully crossed items × templates × orders design, with one observation per cell. Negative estimates are set to 0. Components are reported as shares of total variance.

## 6. Exploratory analyses (labelled as such)
- The share of items whose correctness flips across conditions
- Order invariance of the chosen content
- Accuracy by the correct answer's position
- Fleiss' κ across the three sampled retest runs
- Parse rates
- A sensitivity analysis that excludes unparsed answers

## 7. Things that will not change after data collection
The items, models, templates, orders, scoring rule, hypotheses and tests above. Any deviation will be reported as a deviation.

## References
- Alzahrani, N., et al. (2024). When benchmarks are targets: revealing the sensitivity of LLM leaderboards. *ACL*.
- Brennan, R. L. (2001). *Generalizability Theory*. Springer.
- Hendrycks, D., et al. (2021). Measuring massive multitask language understanding. *ICLR*.
- Pezeshkpour, P., & Hruschka, E. (2024). Large language models sensitivity to the order of options in multiple-choice questions. *Findings of NAACL*.
- Sclar, M., et al. (2024). Quantifying language models' sensitivity to spurious features in prompt design. *ICLR*.
- Weir, J. P. (2005). Quantifying test-retest reliability using the intraclass correlation coefficient and the SEM. *J. Strength Cond. Res.*, 19(1).
- Zheng, C., et al. (2024). Large language models are not robust multiple choice selectors. *ICLR*.

## Note added after data collection
OSF record: https://osf.io/h7kpy. It was submitted after the analysis had been run, because of a miscommunication about whether it had already been submitted. This file, frozen in commit `a1e207b` before data collection, is the time-stamped pre-registration, and its content was not changed.
