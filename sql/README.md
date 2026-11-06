# Snowflake SQL prototypes

Two historical implementations precede the Databricks workflow. They detect commentary, construct within-report windows, call Cortex models, and aggregate available scores. They are reference implementations, not migrations to run against a production warehouse.

- `01_commentary_and_sentiment.sql`: expects an `EXAMPLE_SENTENCES` table with `c1` (sentence number) and `c2` (text), including repeated header rows that delimit reports.
- `02_optimized_sentiment.sql`: expects `EXAMPLE_VALIDATION` with `c1` (document ID), `c2` (sentence number), and `c3` (text).

Provide synthetic input and your own Snowflake account. Cortex calls require model access and incur charges. The historical `SEQ8()` ordering is not a substitute for a persisted source-row key: establish deterministic input ordering before using these prototypes for evaluation. No SQL was executed during cleanup.
