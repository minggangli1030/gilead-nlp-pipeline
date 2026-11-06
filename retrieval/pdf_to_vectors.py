import os
import uuid
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pypdf import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBED_MODEL = os.environ["EMBED_MODEL"]
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

def chunk_text(s: str, w: int, o: int) -> List[str]:
    out, i = [], 0
    if not s:
        return out
    while i < len(s):
        out.append(s[i:i+w])
        step = max(1, w - o)
        i += step
    return [t.strip() for t in out if t and t.strip()]

def extract_rows_from_pdf(pdf_path: Path) -> List[Dict]:
    rows = []
    doc = pdf_path.stem
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"[WARN] Could not read {pdf_path.name}: {e}")
        return rows
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"[WARN] Page {i} in {pdf_path.name} extraction failed: {e}")
            text = ""
        for j, c in enumerate(chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP), 1):
            rows.append({
                "id": f"{doc}-p{i}-{j}",
                "doc": doc,
                "page": i,
                "text": c
            })
    return rows

def main():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {INPUT_DIR}. Drop files there first.")
        return

    all_rows = []
    for p in pdfs:
        extracted = extract_rows_from_pdf(p)
        print(f"{p.name}: extracted {len(extracted)} chunks")
        all_rows.extend(extracted)

    if not all_rows:
        print("No text extracted. If your PDFs are scanned, add OCR.")
        return

    # Embed in batches
    texts = [r["text"] for r in all_rows]
    embeddings = []
    B = 96
    for k in range(0, len(texts), B):
        batch = texts[k:k+B]
        resp = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=batch,
            parameters={"input_type": "passage", "truncate": "END"}  # safe defaults
        )
        embeddings.extend([d.values for d in resp.data])
        print(f"Embedded {min(k+B, len(texts))}/{len(texts)}")

    # Save to Parquet
    df = pd.DataFrame(all_rows)
    df["embedding"] = embeddings
    pq.write_table(pa.Table.from_pandas(df), OUTPUT_DIR / "pdf_vectors.parquet")
    # Also save meta and FAISS index
    meta_records = df.drop(columns=["embedding"]).to_dict("records")
    with open(OUTPUT_DIR / "pdf_vectors.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_records, f, ensure_ascii=False, indent=2)

    # Build a FAISS index with normalized vectors
    try:
        import numpy as np, faiss
        X = np.vstack(df["embedding"].to_numpy()).astype("float32")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        idx = faiss.IndexFlatIP(X.shape[1])
        idx.add(X)
        faiss.write_index(idx, str(OUTPUT_DIR / "pdf_vectors.faiss"))
        print(f"Wrote FAISS index to {OUTPUT_DIR/'pdf_vectors.faiss'}")
    except Exception as e:
        print(f"[WARN] FAISS index not written: {e}")

    print(f"Saved {len(df)} rows to {OUTPUT_DIR/'pdf_vectors.parquet'}")

if __name__ == "__main__":
    main()
