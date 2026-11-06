# Gilead · Financial Document NLP Pipeline

**Author: Minggang Li** · Fall 2025 project · Python, NLP, Databricks, Snowflake

An auditable document ETL and semantic-retrieval project: turning financial research PDFs into structured sentences, dense-vector search, and comparable model scores. I implemented the Fall 2025 code in the original team project: PDF extraction and cleaning, sentiment-model integration, context-window and few-shot experiments, SQL prototypes, and a retrieval prototype.

This standalone portfolio adaptation consolidates that work into one primary parsing path, one end-to-end scoring notebook, and two focused experiment notebooks. Original reports, extracted text, result exports, credentials, and vector indexes are excluded. The company references and examples in executable notebooks are generalized or fictional.

## Project outcomes

- **852 research reports** processed in the original project workflow.
- **Sentiment-scoring RMSE: 0.80 → 0.66**, a 17.5% relative reduction.
- Dense-vector retrieval with normalized inner-product ranking in FAISS, alongside context-window and multi-model sentiment experiments.

The report count and RMSE are author-reported aggregate outcomes. Evaluation data and the original benchmark are excluded; this public adaptation does not reproduce those figures or isolate the contribution of retrieval versus ensemble scoring.

## What the code demonstrates

- **Document engineering:** reconstruct PDF text, remove table-like noise, split sentences, and retain document and sentence identifiers.
- **Model integration:** compare a financial classifier (FinBERT) with generative sentiment scoring through Claude and a Databricks-hosted Llama endpoint.
- **Experiment design:** vary context windows (1, 3, 5, 7 sentences) and generative-model examples (zero through three shots).
- **Data platforms:** implement commentary detection and sentiment aggregation with Snowflake Cortex SQL; run the later workflow in Databricks.
- **Retrieval:** chunk PDFs, request embeddings, and search a local FAISS index.

## Workflow

```text
PDFs → text cleaning → ordered sentences + metadata → document-bounded context
     → FinBERT / Claude / Llama scoring → aligned comparison tables
```

The retrieval prototype is a separate exploration, not a dependency of sentiment scoring.

## Start here

| Path | Purpose |
| --- | --- |
| [scripts/parse_documents.py](scripts/parse_documents.py) | Primary batch parser; does not move source PDFs |
| [src/pdf_parser.py](src/pdf_parser.py) | Shared extraction, cleaning, sentence splitting, and metadata logic |
| [notebooks/01_sentiment_pipeline.ipynb](notebooks/01_sentiment_pipeline.ipynb) | Five-sentence context and three-model scoring |
| [notebooks/02_context_window_comparison.ipynb](notebooks/02_context_window_comparison.ipynb) | Context-window experiment |
| [notebooks/03_few_shot_comparison.ipynb](notebooks/03_few_shot_comparison.ipynb) | Claude/Llama few-shot experiment |
| [docs/METHODS.md](docs/METHODS.md) | Design choices and limitations |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | Authorship and consolidation map |
| [sql/](sql/) | Earlier Snowflake implementation |
| [retrieval/](retrieval/) | Optional PDF similarity-search prototype |

## Run the parser locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/parse_documents.py /path/to/2025_Q4/example_publisher data/parsed.csv
```

Run the offline checks with `python -m pip install pytest` and `python -m pytest -q tests`. They generate a fictional PDF and verify parsing, identifiers, metadata, and source-file preservation.

Create the output directory first (`mkdir -p data`). Supply PDFs you have permission to process. Folder names such as `2025_Q4` allow year and quarter extraction. Sentence tokenization falls back to regular expressions if NLTK resources are unavailable. Scanned documents require an OCR step, which is not included.

The CSV contains `Global #`, `Doc #`, `Sentence #`, `Firm name`, `Year`, `Quarter`, `File title`, `Doc published date`, and `Context`.

## Run the sentiment notebooks

Import the notebooks into your own Databricks workspace. Install the packages listed in `requirements-notebooks.txt` using that runtime's supported package workflow. Supply `CLAUDE_API_KEY` through your environment or secret manager, enable an authorized Llama serving endpoint, and replace the example model settings with models available to your account.

Load the parsed CSV into a Spark table named `parsed_data` for notebook 01. The experiment notebooks expect a `14_docs` input table; check their loading cells and column aliases against your own synthetic input. They are independent research notebooks, not a sequential job. Output cells are intentionally cleared. No cloud inference or paid API calls were run for this portfolio preparation.

## Results and limits

This repository showcases implementation and experiment structure. It includes no original project data or model-comparison outputs. The aggregate outcomes above describe the original project, while the included code demonstrates its approach. Synthetic examples are illustrative rather than benchmark labels. FinBERT polarity is mapped onto a sentiment scale; it is not calibrated to the generative models' scores. See [methods](docs/METHODS.md) before interpreting an output.

## Attribution and release

Developed in the Gilead Discovery team project; this repository presents my Fall 2025 implementation. No employer endorsement is implied. Dependencies and pretrained models retain their own terms. The author confirmed this code may be published with original data removed. No new project-wide license is assigned here.
