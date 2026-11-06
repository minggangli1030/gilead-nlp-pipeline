import os
import sys
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone

# Use the SAME embedding model as ingestion for consistency
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBED_MODEL = os.environ["EMBED_MODEL"]

BASE_DIR = os.path.dirname(__file__)
OUT_PARQUET = os.path.join(BASE_DIR, "output", "pdf_vectors.parquet")
OUT_FAISS = os.path.join(BASE_DIR, "output", "pdf_vectors.faiss")
OUT_META = os.path.join(BASE_DIR, "output", "pdf_vectors.meta.json")

def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n==0] = 1.0
    return v / n

def embed_query(texts):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    resp = pc.inference.embed(model=EMBED_MODEL, inputs=texts,
                              parameters={"input_type": "query", "truncate": "END"})
    return np.array([d.values for d in resp.data], dtype="float32")

def search(query: str, k: int = 5):
    import faiss
    # load data
    df = pd.read_parquet(OUT_PARQUET)
    with open(OUT_META, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # load FAISS index
    index = faiss.read_index(OUT_FAISS)

    # embed query using same model
    vq = embed_query([query]).astype("float32")
    vq = normalize(vq)
    D, I = index.search(vq, k)

    out = []
    for d, i in zip(D[0], I[0]):
        if i < 0 or i >= len(meta):
            continue
        m = meta[int(i)]
        out.append({
            "score": float(d),
            "doc": m["doc"],
            "page": m["page"],
            "id": m["id"],
            "text": m["text"][:400]
        })
    return out

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = input("Query: ").strip()
    results = search(q, k=5)
    for r in results:
        print(f"{r['score']:.3f} | {r['doc']} p{r['page']} | {r['id']}\n{r['text']}\n")
