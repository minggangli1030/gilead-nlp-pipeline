# Optional PDF retrieval prototype

Chunks authorized PDFs, embeds passages with Pinecone inference, and stores local Parquet metadata plus a normalized FAISS index. Query vectors use the same model with query input type. No original PDFs, text metadata, embeddings, or indexes are distributed.

```bash
python -m pip install -r requirements-retrieval.txt
cp .env.example .env
mkdir -p retrieval/input
# Set PINECONE_API_KEY and a Pinecone-supported EMBED_MODEL in .env.
# Add your own authorized PDFs under retrieval/input/.
python retrieval/pdf_to_vectors.py
python retrieval/query_local.py "example query"
```

Requires a model supporting the `passage` and `query` input types used by these scripts. The historical embedding-model default was removed because it did not establish Pinecone compatibility. Hosted integration has not been exercised during preparation. Calls send document text to the embedding provider. Use synthetic or otherwise authorized documents.
