# Authorship and consolidation

Minggang Li confirmed authorship of all code under `Fa25_Code` in the original Gilead Discovery team repository. Scope reviewed: **Fa25_Code only**. No Spring 2026 code or data from other repository folders is included.

| Original area | Portfolio location | Decision |
| --- | --- | --- |
| Final Pipeline/final_parsing.py | scripts/parse_documents.py | Canonical batch entry point; imports shared parser |
| PyPDF/scripts/parse_pdfs.py | src/pdf_parser.py | Canonical extraction/cleaning dependency |
| Final Pipeline/rate_sentiment.ipynb | notebooks/01_sentiment_pipeline.ipynb | End-to-end scoring; removed embedded key |
| three_models_window_size_testing.ipynb | notebooks/02_context_window_comparison.ipynb | Representative three-model window experiment |
| two_models_few_shot_testing.ipynb | notebooks/03_few_shot_comparison.ipynb | Representative generative few-shot experiment |
| Snowflakes SQL | sql/ | Retained two distinct prototypes with generic input table names |
| Pinecone ingestion/query scripts | retrieval/ | Retained source only; corrected inconsistent query API and Parquet export |
| Source-specific parsers and merge scripts | Omitted | Duplicate/hardcoded report paths; canonical parser represents the reusable work |
| Setup notebooks and notebook generator | Omitted | Superseded by selected notebooks and setup documentation |
| Single-model comparison notebooks | Omitted | Overlap with selected multi-model experiments |
| CSV/PDF/vector artifacts, test-output.py | Omitted | Source text, source-specific examples, result exports, or generated index contents |
| Original README/context/agent notes | Rewritten or omitted | Replaced by public-facing documentation |

Notebook outputs, execution counts, attachments, and workspace metadata are cleared. Original report-derived few-shot snippets are replaced with fictional examples. No original Git history is copied. No performance metrics are inferred from the withheld outputs.
