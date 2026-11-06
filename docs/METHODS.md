# Methods and interpretation

## Parsing

The primary parser recursively processes PDFs, reconstructs wrapped text, filters probable tables, splits sentences, and writes stable row/document/sentence identifiers. Deduplication is within each document. Heuristics may remove meaningful text or retain noise; this is not a layout-aware or OCR parser. The shared module exposes reusable helpers; its historical folder-moving CLI was removed. The documented batch entry point leaves source PDFs in place.

## Scoring

The final notebook uses two preceding and two following sentences around the target sentence, bounded by document. FinBERT probabilities are converted using `4 + 3 * (p_positive - p_negative)`, clipped to 1–7. Generative prompts reserve 0 for objective/non-commentary and 1–7 for sentiment. These scales are not interchangeable calibrated measurements.

The original JSON fallback returns a neutral score of 4 for unparseable model responses. That can conceal failures and bias averages. Before using this in an evaluation, separate parse failures, timeouts, and missing outputs from valid neutral scores. Row-count equality alone does not establish correct alignment or semantic validity.

## Experiments

Window comparison: 1, 3, 5, and 7 sentences, with exploratory row limits. Few-shot comparison: zero, one, two, and three synthetic examples for Claude and Llama. FinBERT is a classifier and is not included as a few-shot generative model. Neither experiment notebook is a held-out accuracy benchmark.

## Remaining engineering work

- Validate label definitions and inter-annotator agreement on an authorized dataset.
- Freeze grouped train/development/test partitions by document before model selection.
- Validate score bounds, JSON schema, error rates, and per-model row identity.
- Pin and test runtime/model versions; hosted endpoints and model availability change.
- Measure latency, cost, retries, and document-level robustness.

The portfolio cleanup does not imply that these follow-up experiments have been performed.

## Aggregate outcome provenance

The author supplied the original project report count (852) and sentiment-scoring RMSE change (0.80 to 0.66). Those values were not established by the reviewed Fa25_Code source files or reproduced during cleanup. No claim is made that this FAISS prototype alone caused the RMSE change, that a learned reranker is included, or that production latency was measured. The retrieval implementation uses exact inner-product search after normalization; it is not a distributed index or a deployed low-latency service.
