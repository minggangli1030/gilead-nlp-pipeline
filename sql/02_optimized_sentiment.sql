-- Portfolio reference: provide your own synthetic input table; see sql/README.md.
-- Optimized, reduced models to Llama2-70b-chat, mixtral-8x7b
/* =========================
   0) SOURCE + ORDER + CLEAN
   ========================= */
CREATE OR REPLACE TEMP TABLE SRC AS
WITH raw AS (
  SELECT SEQ8() AS row_idx, c1, c2, c3
  FROM EXAMPLE_VALIDATION
),
marked AS (
  SELECT
    row_idx, c1, c2, c3,
    IFF(LOWER(c1)='doc #' AND LOWER(c2)='sentence #' AND LOWER(c3)='content', 1, 0) AS is_header
  FROM raw
),
filtered AS (
  SELECT
    row_idx,
    TRY_TO_NUMBER(c1)::INT AS report_id,
    TRY_TO_NUMBER(c2)::INT AS sentence_number,
    c3::STRING AS sentence
  FROM marked
  WHERE is_header = 0
)
SELECT
  report_id, row_idx, sentence_number, sentence
FROM filtered
WHERE report_id IS NOT NULL
;

/* =========================================
   1) NEIGHBORS (only what we actually need)
   ========================================= */
CREATE OR REPLACE TEMP TABLE NEI AS
SELECT
  report_id,
  row_idx,
  sentence_number,
  LEFT(sentence, 2000) AS n_text,
  LEFT(LAG(sentence)  OVER (PARTITION BY report_id ORDER BY row_idx), 2000) AS n_minus1_text,
  LEFT(LEAD(sentence) OVER (PARTITION BY report_id ORDER BY row_idx), 2000) AS n_plus1_text
FROM SRC
;

/* ======================================================
   2) COMMENTARY SCORES (mistral-large, n-1 / n / n+1)
      + Add a boolean flag SENTENCE_IS_COMMENTARY (COMM_N>0.33)
   ====================================================== */
CREATE OR REPLACE TEMP TABLE COMM AS
WITH comm_prompt AS (
  SELECT
    'You are a financial text *commentary* detector. Return ONLY strict JSON: ' ||
    '{"score": number in [0,1]} ' ||
    'Where score=0 means NOT commentary, score=1 means pure commentary. ' ||
    'Text: ' AS prompt_prefix
),
win_rows AS (
  SELECT
    n.report_id,
    n.row_idx,
    n.sentence_number,
    v.win_id,
    CASE v.win_id
      WHEN 'n-1' THEN n.n_minus1_text
      WHEN 'n'   THEN n.n_text
      WHEN 'n+1' THEN n.n_plus1_text
    END AS txt
  FROM NEI n
  JOIN (SELECT * FROM (VALUES ('n-1'), ('n'), ('n+1')) v(win_id)) v
    ON TRUE
)
SELECT
  report_id,
  row_idx,
  sentence_number,
  win_id,
  TRY_TO_DOUBLE(TO_VARCHAR(
    TRY_PARSE_JSON(
      SNOWFLAKE.CORTEX.COMPLETE(
        'mistral-large',
        (SELECT prompt_prefix FROM comm_prompt) || COALESCE(txt, '')
      )
    ):"score"
  )) AS comm_score
FROM win_rows
WHERE txt IS NOT NULL
;

CREATE OR REPLACE TEMP TABLE COMM_FLAT AS
SELECT
  c.report_id,
  c.row_idx,
  c.sentence_number,
  MAX(IFF(win_id='n-1', comm_score, NULL)) AS comm_n_minus1,
  MAX(IFF(win_id='n',   comm_score, NULL)) AS comm_n,
  MAX(IFF(win_id='n+1', comm_score, NULL)) AS comm_n_plus1
FROM COMM c
GROUP BY 1,2,3
;

/* Rolling commentary + boolean flag (renamed to SENTENCE_IS_COMMENTARY) */
CREATE OR REPLACE TEMP TABLE COMM_ROLL AS
SELECT
  f.report_id,
  f.row_idx,
  f.sentence_number,
  f.comm_n                                                     AS COMM_N,
  CASE WHEN f.comm_n IS NOT NULL AND f.comm_n_minus1 IS NOT NULL
       THEN (f.comm_n + f.comm_n_minus1)/2 END                AS COMM_N_NMINUS1,
  CASE WHEN f.comm_n IS NOT NULL AND f.comm_n_plus1  IS NOT NULL
       THEN (f.comm_n + f.comm_n_plus1)/2 END                 AS COMM_N_NPLUS1,
  /* Commentary flag (threshold 0.33) - renamed from COMM_IS_COMMENTARY */
  (f.comm_n > 0.33)::BOOLEAN                                  AS SENTENCE_IS_COMMENTARY
FROM COMM_FLAT f
;

/* ======================================================
   3) SENTIMENT (llama2-70b-chat, mixtral-8x7b) for ALL rows
      Windows: n-1&n, n, n&n+1 (skip edges only for paired windows)
      Robust parse: JSON -> regex on "score" -> first number
   ====================================================== */
CREATE OR REPLACE TEMP TABLE SENT_RAW AS
WITH p_sent AS (
  SELECT
    'Financial sentiment. JSON: {"score": 0-5, "label": "optimistic"|"pessimistic"|"neutral", "reason": <25 words}. ' ||
    '0=pessimistic, 2.5=neutral, 5=optimistic. Text: ' AS prompt_prefix
),
to_score AS (
  SELECT
    n.report_id,
    n.row_idx,
    n.sentence_number,
    v.win_id,
    CASE v.win_id
      WHEN 'n-1&n' THEN n.n_minus1_text || ' ' || n.n_text
      WHEN 'n'     THEN n.n_text
      WHEN 'n&n+1' THEN n.n_text || ' ' || n.n_plus1_text
    END AS txt
  FROM NEI n
  JOIN (SELECT column1 AS win_id FROM VALUES ('n-1&n'), ('n'), ('n&n+1')) v ON TRUE
  WHERE (v.win_id <> 'n-1&n' OR n.n_minus1_text IS NOT NULL)
    AND (v.win_id <> 'n&n+1' OR n.n_plus1_text  IS NOT NULL)
),
-- Core helper: run a model and robustly extract a numeric score
model_out AS (
  SELECT
    t.report_id, t.row_idx, t.sentence_number, t.win_id,
    'llama2-70b-chat'::STRING AS model_name,
    SNOWFLAKE.CORTEX.COMPLETE(
      'llama2-70b-chat',
      (SELECT prompt_prefix FROM p_sent) || COALESCE(t.txt, '')
    ) AS raw_out
  FROM to_score t
  UNION ALL
  SELECT
    t.report_id, t.row_idx, t.sentence_number, t.win_id,
    'mixtral-8x7b'::STRING AS model_name,
    SNOWFLAKE.CORTEX.COMPLETE(
      'mixtral-8x7b',
      (SELECT prompt_prefix FROM p_sent) || COALESCE(t.txt, '')
    ) AS raw_out
  FROM to_score t
),
parsed AS (
  SELECT
    report_id, row_idx, sentence_number, win_id, model_name, raw_out,
    TRY_TO_DOUBLE(TO_VARCHAR(TRY_PARSE_JSON(raw_out):"score")) AS score_json,

    /* Grab the number following "score": using a capturing group (group 1) */
    TRY_TO_DOUBLE(
      REGEXP_SUBSTR(
        raw_out,
        '"score"\\s*:\\s*(-?[0-9]+(\\.[0-9]+)?)',
        1, 1, 'i', 1
      )
    ) AS score_rx_named,

    /* Fallback: first number in the output (no groups needed) */
    TRY_TO_DOUBLE(
      REGEXP_SUBSTR(
        raw_out,
        '-?[0-9]+(\\.[0-9]+)?'
      )
    ) AS score_rx_first
  FROM model_out
)
SELECT
  report_id, row_idx, sentence_number, win_id, model_name,
  COALESCE(score_json, score_rx_named, score_rx_first) AS model_score
FROM parsed
;

/* ============================
   3.5) PIVOT SENTIMENT SCORES
   ============================ */
CREATE OR REPLACE TEMP TABLE SENT_PIVOT AS
SELECT
  report_id,
  sentence_number,
  -- llama2-70b-chat scores
  MAX(IFF(model_name='llama2-70b-chat' AND win_id='n-1&n', model_score, NULL)) AS llama2_n_nminus1,
  MAX(IFF(model_name='llama2-70b-chat' AND win_id='n',     model_score, NULL)) AS llama2_n,
  MAX(IFF(model_name='llama2-70b-chat' AND win_id='n&n+1', model_score, NULL)) AS llama2_n_nplus1,
  -- mixtral-8x7b scores
  MAX(IFF(model_name='mixtral-8x7b' AND win_id='n-1&n', model_score, NULL)) AS mixtral8x7b_n_nminus1,
  MAX(IFF(model_name='mixtral-8x7b' AND win_id='n',     model_score, NULL)) AS mixtral8x7b_n,
  MAX(IFF(model_name='mixtral-8x7b' AND win_id='n&n+1', model_score, NULL)) AS mixtral8x7b_n_nplus1
FROM SENT_RAW
GROUP BY report_id, sentence_number
;

/* ============================
   3.6) CALCULATE AVERAGES
   ============================ */
CREATE OR REPLACE TEMP TABLE SENT_AVG AS
SELECT
  report_id,
  sentence_number,
  -- Average across both models for each window
  (NVL(llama2_n_nminus1, 0) + NVL(mixtral8x7b_n_nminus1, 0)) / 
    NULLIFZERO((llama2_n_nminus1 IS NOT NULL)::INT + (mixtral8x7b_n_nminus1 IS NOT NULL)::INT) AS avg_nminus1,
  
  (NVL(llama2_n, 0) + NVL(mixtral8x7b_n, 0)) / 
    NULLIFZERO((llama2_n IS NOT NULL)::INT + (mixtral8x7b_n IS NOT NULL)::INT) AS avg_n,
  
  (NVL(llama2_n_nplus1, 0) + NVL(mixtral8x7b_n_nplus1, 0)) / 
    NULLIFZERO((llama2_n_nplus1 IS NOT NULL)::INT + (mixtral8x7b_n_nplus1 IS NOT NULL)::INT) AS avg_nplus1
FROM SENT_PIVOT
;

/* ============================
   4) FINAL SHAPE (exact order)
      KEEPS ALL ROWS - commentary and non-commentary
   ============================ */
CREATE OR REPLACE VIEW FINAL_OUTPUT AS
WITH base AS (
  SELECT
    s.report_id,
    s.row_idx,
    s.sentence_number,
    '"' || REPLACE(s.sentence, '"', '""') || '"' AS sentence
  FROM SRC s
),
c AS ( SELECT * FROM COMM_ROLL ),
p AS ( SELECT * FROM SENT_PIVOT ),
a AS ( SELECT * FROM SENT_AVG )
SELECT
  b.report_id                                              AS REPORT_ID,
  b.row_idx                                                AS ROW_IDX,
  b.sentence_number                                        AS SENTENCE_NUMBER,
  b.sentence                                               AS SENTENCE,

  -- commentary columns
  c.COMM_N_NMINUS1                                         AS COMM_N_NMINUS1,
  c.COMM_N                                                 AS COMM_N,
  c.COMM_N_NPLUS1                                          AS COMM_N_NPLUS1,
  c.SENTENCE_IS_COMMENTARY                                 AS SENTENCE_IS_COMMENTARY,

  -- model scores
  p.llama2_n_nminus1                                       AS LLAMA2_N_NMINUS1,
  p.llama2_n                                               AS LLAMA2_N,
  p.llama2_n_nplus1                                        AS LLAMA2_N_NPLUS1,
  p.mixtral8x7b_n_nminus1                                  AS MIXTRAL8X7B_N_NMINUS1,
  p.mixtral8x7b_n                                          AS MIXTRAL8X7B_N,
  p.mixtral8x7b_n_nplus1                                   AS MIXTRAL8X7B_N_NPLUS1,

  -- across-model averages
  a.avg_nminus1                                            AS AVG_NMINUS1,
  a.avg_n                                                  AS AVG_N,
  a.avg_nplus1                                             AS AVG_NPLUS1,

  -- final average over available window avgs
  (
    NVL(a.avg_nminus1,0) + NVL(a.avg_n,0) + NVL(a.avg_nplus1,0)
  ) / NULLIFZERO(
        (a.avg_nminus1 IS NOT NULL)::INT
      + (a.avg_n IS NOT NULL)::INT
      + (a.avg_nplus1 IS NOT NULL)::INT
     )                                                     AS SENTENCE_FINAL_AVG
FROM base b
JOIN c
  ON c.report_id = b.report_id
 AND c.sentence_number = b.sentence_number
LEFT JOIN p
  ON p.report_id = b.report_id
 AND p.sentence_number = b.sentence_number
LEFT JOIN a
  ON a.report_id = b.report_id
 AND a.sentence_number = b.sentence_number
-- NO WHERE CLAUSE - keeps all rows regardless of commentary flag
ORDER BY REPORT_ID, SENTENCE_NUMBER
;

-- Materialize / inspect
SELECT * FROM FINAL_OUTPUT;
