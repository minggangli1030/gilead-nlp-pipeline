-- Portfolio reference: provide your own synthetic input table; see sql/README.md.
/* ==========================================================
   0) SOURCE + ORDER + REPORT BOUNDARIES
   ========================================================== */
CREATE OR REPLACE TEMP TABLE SRC AS
WITH raw AS (
  SELECT SEQ8() AS row_idx, c1, c2
  FROM EXAMPLE_SENTENCES
),
marked AS (
  SELECT
    row_idx, c1, c2,
    IFF(LOWER(c1)='sentence_number' AND LOWER(c2)='sentence', 1, 0) AS is_header
  FROM raw
),
with_reports AS (
  SELECT
    row_idx, c1, c2,
    SUM(is_header) OVER (ORDER BY row_idx ROWS UNBOUNDED PRECEDING) AS report_id
  FROM marked
)
SELECT
  report_id,
  row_idx,
  TRY_TO_NUMBER(c1)::INT AS sentence_number,
  c2::STRING AS sentence
FROM with_reports
WHERE NOT (LOWER(c1)='sentence_number' AND LOWER(c2)='sentence')
;

/* ==========================================================
   1) COMMENTARY PASS (decides gating)
      Windows we'll later score: n-1 | n-1&n | n | n&n+1 | n+1
      We gate **by the 'n' window only** (clean, per-sentence decision).
   ========================================================== */
CREATE OR REPLACE TEMP TABLE COMMENTARY_SCORES AS
WITH neighbors AS (
  SELECT
    report_id, row_idx, sentence_number, sentence,
    LAG(sentence)  OVER (PARTITION BY report_id ORDER BY row_idx) AS prev_sentence,
    LEAD(sentence) OVER (PARTITION BY report_id ORDER BY row_idx) AS next_sentence
  FROM SRC
),
windows AS (
  -- build the 5 windows with their availability
  SELECT report_id, row_idx, sentence_number, 'n-1'    AS window_id, prev_sentence                                   AS window_text FROM neighbors WHERE prev_sentence IS NOT NULL
  UNION ALL
  SELECT report_id, row_idx, sentence_number, 'n-1&n'  AS window_id, (prev_sentence || ' ' || sentence)               AS window_text FROM neighbors WHERE prev_sentence IS NOT NULL
  UNION ALL
  SELECT report_id, row_idx, sentence_number, 'n'      AS window_id, sentence                                        AS window_text FROM neighbors
  UNION ALL
  SELECT report_id, row_idx, sentence_number, 'n&n+1'  AS window_id, (sentence || ' ' || next_sentence)               AS window_text FROM neighbors WHERE next_sentence IS NOT NULL
  UNION ALL
  SELECT report_id, row_idx, sentence_number, 'n+1'    AS window_id, next_sentence                                   AS window_text FROM neighbors WHERE next_sentence IS NOT NULL
),
comm_prompt AS (
  SELECT
    'You are a financial text commentary detector. Return ONLY strict JSON: ' ||
    '{"score": number in [0,1]} ' ||
    'Definition: 0 = objective (NOT a comment); 1 = subjective/opinionated (IS a comment). ' ||
    'Text: ' AS prompt_prefix
)
SELECT
  w.report_id,
  w.row_idx,
  w.sentence_number,
  w.window_id,
  w.window_text,
  TRY_TO_DOUBLE(TO_VARCHAR(
    TRY_PARSE_JSON(
      SNOWFLAKE.CORTEX.COMPLETE(
        'mixtral-8x7b',
        (SELECT prompt_prefix FROM comm_prompt) || w.window_text
      )
    ):"score"
  )) AS commentary_score
FROM windows w
;

/* Per-sentence keep flag: use the 'n' window commentary */
CREATE OR REPLACE TEMP TABLE COMMENTARY_KEEP AS
SELECT
  report_id, sentence_number,
  MAX(IFF(window_id='n', commentary_score, NULL)) AS comm_n,
  IFF(MAX(IFF(window_id='n', commentary_score, NULL)) > 0.33, 1, 0) AS keep_sentence  -- 1 = commentary; 0 = NOT commentary
FROM COMMENTARY_SCORES
GROUP BY report_id, sentence_number
;

/* ==========================================================
   2) SENTIMENT (2 MODELS) — RUN ONLY ON KEPT SENTENCES
      Scale: 0 = most pessimistic, 2.5 = neutral, 5 = most optimistic
   ========================================================== */
CREATE OR REPLACE TEMP TABLE BMO_SENTIMENTS_PER_MODEL AS
WITH p_sent AS (
  SELECT
    'You are a financial sentiment evaluator. Respond ONLY in strict JSON format: ' ||
    '{"score": float ∈ [0,5], where 0 = most pessimistic, 2.5 = neutral, and 5 = most optimistic, ' ||
    '"label": "optimistic" | "pessimistic" | "neutral", ' ||
    '"reason": string (≤25 words)}. ' ||
    'Text: ' AS prompt_prefix
),
to_score AS (  -- only windows from sentences we keep
  SELECT c.*
  FROM COMMENTARY_SCORES c
  JOIN COMMENTARY_KEEP k
    ON k.report_id = c.report_id AND k.sentence_number = c.sentence_number
  WHERE k.keep_sentence = 1
),
m1 AS (
  SELECT t.*, 'llama2-70b-chat'::STRING AS model_name,
         TRY_PARSE_JSON(SNOWFLAKE.CORTEX.COMPLETE('llama2-70b-chat', (SELECT prompt_prefix FROM p_sent) || t.window_text)) AS j
  FROM to_score t
),
m4 AS (
  SELECT t.*, 'mixtral-8x7b'::STRING AS model_name,
         TRY_PARSE_JSON(SNOWFLAKE.CORTEX.COMPLETE('mixtral-8x7b', (SELECT prompt_prefix FROM p_sent) || t.window_text)) AS j
  FROM to_score t
),
per_model_json AS (
  SELECT * FROM m1
  UNION ALL SELECT * FROM m4
),
extracted AS (
  SELECT
    report_id,
    row_idx,
    sentence_number,
    window_id,
    model_name,
    TRY_TO_DOUBLE(TO_VARCHAR(j:"score")) AS model_score  -- [0,5]
  FROM per_model_json
  WHERE j IS NOT NULL
)
SELECT
  e.report_id,
  e.sentence_number,
  s.sentence,
  e.window_id,
  e.model_name,
  e.model_score
FROM extracted e
JOIN SRC s
  ON s.report_id = e.report_id AND s.row_idx = e.row_idx
ORDER BY report_id, sentence_number, window_id, model_name
;

/* ==========================================================
   3) ACROSS-MODEL WINDOW AVERAGES (for kept sentences only)
   ========================================================== */
CREATE OR REPLACE TEMP TABLE BMO_SENTIMENTS AS
SELECT
  report_id,
  sentence_number,
  window_id,
  AVG(model_score) AS window_avg_score
FROM BMO_SENTIMENTS_PER_MODEL
GROUP BY report_id, sentence_number, window_id
;



/* ==========================================================
   4) FLAT VIEW
      - All SRC columns
      - 10 per-model columns (2 models × 5 windows)
        * If sentence NOT commentary → **-1** in all 10 columns
        * If edge window unavailable → NULL (only for first/last lines)
      - 5 window means (across models). If NOT commentary → **-1**
      - final per-sentence mean over available window means (exclude -1)
   ========================================================== */
CREATE OR REPLACE VIEW BMO_SENTIMENTS_FLAT AS
WITH base AS (  -- tall per-model scored (kept sentences)
  SELECT
    p.report_id, p.sentence_number, p.model_name, p.window_id, p.model_score
  FROM BMO_SENTIMENTS_PER_MODEL p
),
k AS (          -- keep flag per sentence
  SELECT report_id, sentence_number, keep_sentence
  FROM COMMENTARY_KEEP
),
-- availability flags to distinguish edge-window NULLs from non-commentary -1
avail AS (
  SELECT
    s.report_id, s.sentence_number,
    (LAG(1) OVER (PARTITION BY s.report_id ORDER BY s.row_idx))             AS has_prev,
    1                                                                       AS has_n,
    (LEAD(1) OVER (PARTITION BY s.report_id ORDER BY s.row_idx))            AS has_next
  FROM SRC s
),
pivoted AS (  -- 10 per-model columns (NULL if absent; we'll fill -1 for non-commentary later)
  SELECT
    s.report_id,
    s.row_idx,
    s.sentence_number,
    '"' || REPLACE(s.sentence, '"', '""') || '"' AS sentence,

    /* llama2-70b-chat */
    MAX(IFF(model_name='llama2-70b-chat' AND window_id='n-1',   model_score, NULL)) AS llama2_n_minus1,
    MAX(IFF(model_name='llama2-70b-chat' AND window_id='n-1&n', model_score, NULL)) AS llama2_n_minus1_n,
    MAX(IFF(model_name='llama2-70b-chat' AND window_id='n',     model_score, NULL)) AS llama2_n,
    MAX(IFF(model_name='llama2-70b-chat' AND window_id='n&n+1', model_score, NULL)) AS llama2_n_nplus1,
    MAX(IFF(model_name='llama2-70b-chat' AND window_id='n+1',   model_score, NULL)) AS llama2_n_plus1,

    /* mixtral-8x7b */
    MAX(IFF(model_name='mixtral-8x7b' AND window_id='n-1',   model_score, NULL)) AS mixtral8x7b_n_minus1,
    MAX(IFF(model_name='mixtral-8x7b' AND window_id='n-1&n', model_score, NULL)) AS mixtral8x7b_n_minus1_n,
    MAX(IFF(model_name='mixtral-8x7b' AND window_id='n',     model_score, NULL)) AS mixtral8x7b_n,
    MAX(IFF(model_name='mixtral-8x7b' AND window_id='n&n+1', model_score, NULL)) AS mixtral8x7b_n_nplus1,
    MAX(IFF(model_name='mixtral-8x7b' AND window_id='n+1',   model_score, NULL)) AS mixtral8x7b_n_plus1

  FROM SRC s
  LEFT JOIN base b
    ON b.report_id = s.report_id
   AND b.sentence_number = s.sentence_number
  GROUP BY s.report_id, s.row_idx, s.sentence_number, s.sentence
),
means AS (  -- across-model window means (kept sentences only)
  SELECT
    report_id, sentence_number,
    MAX(IFF(window_id='n-1',   window_avg_score, NULL)) AS avg_n_minus1,
    MAX(IFF(window_id='n-1&n', window_avg_score, NULL)) AS avg_n_minus1_n,
    MAX(IFF(window_id='n',     window_avg_score, NULL)) AS avg_n,
    MAX(IFF(window_id='n&n+1', window_avg_score, NULL)) AS avg_n_nplus1,
    MAX(IFF(window_id='n+1',   window_avg_score, NULL)) AS avg_n_plus1
  FROM BMO_SENTIMENTS
  GROUP BY report_id, sentence_number
)
SELECT
  p.report_id,
  p.row_idx,
  p.sentence_number,
  p.sentence,

  /* For NON-commentary sentences → -1 in all 10 per-model columns */
  /* llama2-70b-chat */
  IFF(k.keep_sentence=0, -1, p.llama2_n)            AS llama2_n,
  IFF(k.keep_sentence=0, -1, p.llama2_n_nplus1)     AS llama2_n_nplus1,
  IFF(k.keep_sentence=0, -1, p.llama2_n_plus1)      AS llama2_n_plus1,
  IFF(k.keep_sentence=0, -1, p.llama2_n_minus1)     AS llama2_n_minus1,
  IFF(k.keep_sentence=0, -1, p.llama2_n_minus1_n)   AS llama2_n_minus1_n,

  /* mixtral-8x7b */
  IFF(k.keep_sentence=0, -1, p.mixtral8x7b_n_minus1)     AS mixtral8x7b_n_minus1,
  IFF(k.keep_sentence=0, -1, p.mixtral8x7b_n_minus1_n)   AS mixtral8x7b_n_minus1_n,
  IFF(k.keep_sentence=0, -1, p.mixtral8x7b_n)            AS mixtral8x7b_n,
  IFF(k.keep_sentence=0, -1, p.mixtral8x7b_n_nplus1)     AS mixtral8x7b_n_nplus1,
  IFF(k.keep_sentence=0, -1, p.mixtral8x7b_n_plus1)      AS mixtral8x7b_n_plus1,

  /* Across-model window means; -1 for non-commentary */
  IFF(k.keep_sentence=0, -1, m.avg_n_minus1)   AS avg_n_minus1,
  IFF(k.keep_sentence=0, -1, m.avg_n_minus1_n) AS avg_n_minus1_n,
  IFF(k.keep_sentence=0, -1, m.avg_n)          AS avg_n,
  IFF(k.keep_sentence=0, -1, m.avg_n_nplus1)   AS avg_n_nplus1,
  IFF(k.keep_sentence=0, -1, m.avg_n_plus1)    AS avg_n_plus1,

  /* Final per-sentence avg over available window means (ignore -1s and NULLs) */
  ROUND(
    ( NVL2(NULLIF(IFF(k.keep_sentence=0, -1, m.avg_n_minus1), -1), NVL(m.avg_n_minus1,0), 0)
    + NVL2(NULLIF(IFF(k.keep_sentence=0, -1, m.avg_n_minus1_n), -1), NVL(m.avg_n_minus1_n,0), 0)
    + NVL2(NULLIF(IFF(k.keep_sentence=0, -1, m.avg_n), -1),        NVL(m.avg_n,0), 0)
    + NVL2(NULLIF(IFF(k.keep_sentence=0, -1, m.avg_n_nplus1), -1), NVL(m.avg_n_nplus1,0), 0)
    + NVL2(NULLIF(IFF(k.keep_sentence=0, -1, m.avg_n_plus1), -1),  NVL(m.avg_n_plus1,0), 0) )
    /
    NULLIFZERO(
        IFF(k.keep_sentence=0, 0,
            (m.avg_n_minus1   IS NOT NULL)::INT +
            (m.avg_n_minus1_n IS NOT NULL)::INT +
            (m.avg_n          IS NOT NULL)::INT +
            (m.avg_n_nplus1   IS NOT NULL)::INT +
            (m.avg_n_plus1    IS NOT NULL)::INT )
    ),
  3) AS sentence_final_avg

FROM pivoted p
LEFT JOIN k
  ON k.report_id = p.report_id AND k.sentence_number = p.sentence_number
LEFT JOIN means m
  ON m.report_id = p.report_id AND m.sentence_number = p.sentence_number
ORDER BY p.report_id, p.sentence_number;
