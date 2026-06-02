# Cloud Storage Setup Guide

How to configure Azure Blob Storage and Qdrant so you can test the cloud
migration with real `.eml` data.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `RAG_EMBEDDING_PROVIDER` | **Yes** | `openai` | `openai` or `lmstudio` |
| `RAG_EMBEDDING_MODEL` | **Yes** | `text-embedding-3-small` | Embedding model name |
| `RAG_EMBEDDING_API_BASE` | LM Studio mode | `https://api.openai.com/v1` | OpenAI-compatible base URL (LM Studio) |
| `RAG_EMBEDDING_API_KEY` | No | `""` | Optional API key for OpenAI-compatible embedding endpoint |
| `RAG_INDEX_BATCH_SIZE` | No | `200` | Number of emails downloaded and indexed per batch |
| `RAG_DOWNLOAD_WORKERS` | No | `8` | Number of parallel Azure Blob downloads per batch |
| `RAG_EMBEDDING_BATCH_SIZE` | No | `100` | Texts per embedding HTTP request (512 recommended for LM Studio on M-series) |
| `RAG_EMBEDDING_NUM_WORKERS` | No | `1` | Embedding worker processes; keep at `1` — values > 1 flood the LM Studio queue and break the progress bar |
| `OPENAI_API_KEY` | OpenAI mode | — | OpenAI key (LLM and/or OpenAI embeddings) |
| `AZURE_STORAGE_CONNECTION_STRING` | Phase 1 | — | Azure Storage account connection string |
| `AZURE_BLOB_CONTAINER` | Phase 1 | `eml-archive` | Blob container holding `.eml` files |
| `AZURE_BLOB_PREFIX` | No | `""` (all blobs) | Scope listing to a subfolder, e.g. `Inbox/` |
| `VECTOR_STORE_PROVIDER` | Phase 2 | `simple` | `simple` (local JSON), `qdrant`, or `pinecone` |
| `QDRANT_URL` | Qdrant mode | — | Qdrant cluster URL (`https://...:6333`) |
| `QDRANT_API_KEY` | Qdrant Cloud | `""` | Qdrant API key (not needed for local Docker) |
| `QDRANT_COLLECTION_NAME` | Qdrant mode | `email-rag` | Collection used for vectors |
| `QDRANT_PREFER_GRPC` | No | `false` | Use gRPC transport for Qdrant client (recommended: `true` for cloud Qdrant) |
| `PINECONE_API_KEY` | Pinecone mode | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone mode | `email-rag` | Name of the Pinecone serverless index |

All variables are read from `.env` (via `python-dotenv`) or from the shell
environment.  Copy the template and fill in your values:

```bash
cp .env.example .env
```

---

## Phase 1 — Azure Blob Storage for `.eml` Files

### 1. Create an Azure Storage Account

1. Go to [Azure Portal → Storage accounts](https://portal.azure.com/#browse/Microsoft.Storage%2FStorageAccounts).
2. Click **+ Create**.
3. Choose a **Resource group** (or create one, e.g. `rag-study-rg`).
4. **Storage account name**: something unique, e.g. `ragstudyemls`.
5. **Region**: pick the region closest to where you run the code (if using
   GitHub Codespaces, `East US` is a good default).
6. **Performance**: Standard.
7. **Redundancy**: LRS (Locally Redundant — cheapest).
8. Click **Review + Create → Create**.

### 2. Create a Blob Container

1. Open the new storage account in the portal.
2. In the left sidebar go to **Data storage → Containers**.
3. Click **+ Container**.
4. **Name**: `eml-archive` (matches the default `AZURE_BLOB_CONTAINER`).
5. **Public access level**: Private.
6. Click **Create**.

### 3. Get the Connection String

1. In the storage account, go to **Security + networking → Access keys**.
2. Click **Show** next to `key1`.
3. Copy the full **Connection string** — it starts with
   `DefaultEndpointsProtocol=https;AccountName=...`.
4. Paste it into your `.env`:

```bash
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=ragstudyemls;AccountKey=XXXXX;EndpointSuffix=core.windows.net
```

### 4. Upload `.eml` Files

From a terminal that has the [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed:

```bash
# Log in (one-time)
az login

# Bulk upload preserving directory structure
az storage blob upload-batch \
  --destination eml-archive \
  --source /path/to/your/eml/root \
  --pattern "*.eml" \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING"
```

> **Tip — large uploads (>1 GB):** Use
> [AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)
> for parallel transfers:
> ```bash
> azcopy copy "/path/to/eml/root" "https://ragstudyemls.blob.core.windows.net/eml-archive" --recursive
> ```

### 5. Verify Upload

```bash
# Quick count of blobs
az storage blob list \
  --container-name eml-archive \
  --connection-string "$AZURE_STORAGE_CONNECTION_STRING" \
  --query "length(@)"
```

### 6. Smoke-Test With the Loader

With your `.env` populated:

```bash
poetry run python -c "
from dotenv import load_dotenv; load_dotenv()
from src.data.loader import load_emails

docs = load_emails(source='azure_blob', num_samples=10)
print(f'Loaded {len(docs)} documents')
for d in docs[:3]:
    print(f'  {d.metadata.get(\"sender\")} — {d.metadata.get(\"subject\")}')
"
```

Expected output: 10 `Document` objects with `sender`, `subject`, and `date`
metadata populated.

---

## Phase 2 — Qdrant Vector Store (Primary)

### 1. Start LM Studio on Host Machine

1. Start LM Studio on your Mac host.
2. Load an embedding model, for example `text-embedding-nomic-embed-text-v1.5`.
3. Enable the OpenAI-compatible server endpoint.
4. Keep the endpoint available at `http://host.docker.internal:1234/v1` for devcontainers.

### 2. Create a Qdrant Cloud Cluster

1. Sign up at <https://cloud.qdrant.io/>.
2. Create a cluster in your preferred region.
3. Copy the cluster URL and API key.

### 3. Configure `.env` for LM Studio + Qdrant Cloud

```bash
RAG_EMBEDDING_PROVIDER=lmstudio
RAG_EMBEDDING_MODEL=text-embedding-bge-m3
RAG_EMBEDDING_API_BASE=http://host.docker.internal:1234/v1
# RAG_EMBEDDING_API_KEY=      # optional
RAG_INDEX_BATCH_SIZE=200
RAG_DOWNLOAD_WORKERS=8
RAG_EMBEDDING_BATCH_SIZE=512   # texts per request; reduce to 128 if LM Studio crashes
RAG_EMBEDDING_NUM_WORKERS=1    # keep at 1 — see note below

VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=https://xxxxxx.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
QDRANT_COLLECTION_NAME=email-rag
QDRANT_PREFER_GRPC=true        # gRPC is significantly faster than REST for bulk upserts
```

### 3a. LM Studio Embedding Smoke Test

Before long indexing jobs, validate one live embedding call:

```bash
poetry run python scripts/smoke_lmstudio_embedding.py
```

Expected output includes:

```text
OK: Embedding OK (dimension=..., model=...)
```

### 4. Small-Scale Test (10 Emails → Qdrant)

This end-to-end test loads 10 emails from Azure, embeds them via LM Studio,
and pushes vectors to Qdrant:

```bash
poetry run python -c "
from dotenv import load_dotenv; load_dotenv()
from src.config.settings import RAGConfig
from src.data.loader import load_emails
from src.storage.persist import StorageManager

RAGConfig.initialize_settings(include_llm=False)

docs = load_emails(source='azure_blob', num_samples=10)
print(f'Loaded {len(docs)} documents from Azure')

StorageManager.create_and_save_index(docs)
print('Index created in Qdrant')
print(f'Collection has vectors: {StorageManager.index_exists()}')
"
```

### 5. Full Batch Indexing (68 K Emails)

```bash
poetry run python scripts/batch_index_to_vector_store.py
```

The script is now provider-aware:
- Uses `VECTOR_STORE_PROVIDER` from `.env`.
- Embeds using configured embedding provider via an **async `IngestionPipeline`** — this sends up to `RAG_EMBEDDING_NUM_WORKERS` concurrent requests to LM Studio and overlaps Qdrant uploads with the next embedding batch.
- Downloads blobs in parallel within each batch (`RAG_DOWNLOAD_WORKERS`).
- Writes `scripts/.vector_batch_checkpoint.txt` after each batch for resume support.

Recommended values for LM Studio on an M-series Mac (set in `.env`):
- `RAG_INDEX_BATCH_SIZE=200`
- `RAG_DOWNLOAD_WORKERS=8`
- `RAG_EMBEDDING_BATCH_SIZE=512`  — larger batches keep the GPU busy for the full inference pass
- `RAG_EMBEDDING_NUM_WORKERS=1`   — **keep at 1**; values > 1 spawn separate processes that flood the LM Studio request queue faster than the GPU can drain it, growing the queue unboundedly and suppressing the progress bar

If LM Studio crashes or the Python process exits with code `139`, reduce `RAG_EMBEDDING_BATCH_SIZE`
first (try `128`, then `64`), then reduce `RAG_INDEX_BATCH_SIZE`.

### 6. Query Against Qdrant

Once vectors are in Qdrant, any query path automatically uses it when
`VECTOR_STORE_PROVIDER=qdrant`:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.query.hybrid import build_hybrid_searcher

searcher = build_hybrid_searcher('your-collection', mode='hybrid')  # reads QDRANT_URL from .env
for ctx in searcher.search_threads('meeting schedule'):
    print(ctx.subject)
"
```

### 7. Resetting Qdrant Before Re-Indexing

To restart indexing from a clean state while preserving collection schema and
payload indexes:

```bash
poetry run python scripts/reset_qdrant_index.py
```

This is the default mode and is recommended when you want to keep field mappings
(e.g., `sender` keyword/text indexes).

To delete the full collection, including schema and payload indexes:

```bash
poetry run python scripts/reset_qdrant_index.py --drop-schema
```

Both modes also remove `scripts/.vector_batch_checkpoint.txt` so batch indexing
starts from the beginning.

## Phase 2b — Pinecone (Optional Legacy Path)

### 1. Create a Free Pinecone Account

1. Sign up at <https://app.pinecone.io/>.
2. The **Starter (free)** plan gives you 1 serverless index, 2 GB storage,
   and ~100 K vectors — enough for 68 K emails.

### 2. Create a Serverless Index

In the Pinecone console:

1. Click **Create Index**.
2. **Index name**: `email-rag` (matches the default `PINECONE_INDEX_NAME`).
3. **Dimensions**: `1536` — must match `text-embedding-3-small` output.
4. **Metric**: `cosine`.
5. **Environment type**: **Serverless**.
6. **Cloud / Region**: AWS `us-east-1` (free-tier region).
7. Click **Create Index**.

### 3. Get the API Key

1. Go to **API Keys** in the Pinecone sidebar.
2. Copy the default key.
3. Add to `.env`:

```bash
PINECONE_API_KEY=pcsk_XXXXXXXXXXXX
PINECONE_INDEX_NAME=email-rag
VECTOR_STORE_PROVIDER=pinecone
```

### 4. Small-Scale Test (10 Emails → Pinecone)

This end-to-end test loads 10 emails from Azure, embeds them, and pushes
vectors to Pinecone:

```bash
poetry run python -c "
from dotenv import load_dotenv; load_dotenv()
from src.config.settings import RAGConfig
from src.data.loader import load_emails
from src.storage.persist import StorageManager

RAGConfig.initialize_settings()

docs = load_emails(source='azure_blob', num_samples=10)
print(f'Loaded {len(docs)} documents from Azure')

index = StorageManager.create_and_save_index(docs)
print('Index created in Pinecone')

# Verify vectors exist
print(f'Index has vectors: {StorageManager.index_exists()}')
"
```

### 5. Full Batch Indexing (68 K Emails)

The batch script processes emails in chunks of 200 with checkpoint
resumption:

```bash
poetry run python scripts/batch_index_to_vector_store.py
```

The script:
- Lists all `.eml` blobs in the Azure container.
- Downloads each batch of 200 to a temporary directory (auto-deleted after
  each batch).
- Parses via `MailArchiveXLoader`, embeds via OpenAI, and upserts to Pinecone.
- Writes `scripts/.vector_batch_checkpoint.txt` after each batch so you
  can safely Ctrl-C and resume later.

> **Batch size note:** The default batch size is 200 emails, tuned for
> memory-constrained environments like GitHub Codespaces (2-core, 4 GB).
> You can increase `BATCH_SIZE` in the script if you have more RAM.

### 5a. Resetting the Index

To wipe all vectors and restart indexing from scratch (e.g., after changing
chunk size or metadata format):

```bash
poetry run python scripts/reset_pinecone_index.py
```

This deletes all vectors from the Pinecone index **and** removes the
checkpoint file so the batch script starts from email 0.

### 5b. Metadata Limits & Safety

Document metadata is automatically trimmed to stay within two hard limits:

| Limit | Source | Value |
|---|---|---|
| Chunk-size ceiling | LlamaIndex sentence splitter | `RAG_CHUNK_SIZE` (default 2048) |
| Per-vector metadata | Pinecone API | 40 960 bytes |

Bulky fields (`source_id`, `to_full`, `cc_full`) are excluded from
embedding/LLM context so they don't count toward the chunk-size ceiling,
but they are still stored in Pinecone for filtering.

To validate these limits locally **before** indexing:

```bash
poetry run pytest tests/test_document_metadata_limits.py -v
```

### 6. Query Against Pinecone (Optional Legacy Path)

Once vectors are in Pinecone, any query path automatically uses it when
`VECTOR_STORE_PROVIDER=pinecone`:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.query.hybrid import build_hybrid_searcher

searcher = build_hybrid_searcher('your-collection', mode='hybrid')
for ctx in searcher.search_threads('meeting schedule'):
    print(ctx.subject)
"
```

---

## Sample `.env` (Both Phases)

```bash
# === Embeddings (Phase 1 default: LM Studio on host) ===
RAG_EMBEDDING_PROVIDER=lmstudio
RAG_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
RAG_EMBEDDING_API_BASE=http://host.docker.internal:1234/v1
# RAG_EMBEDDING_API_KEY=

# Optional if using OpenAI embeddings/LLM instead of LM Studio
# OPENAI_API_KEY=sk-...

# === Phase 1: Azure Blob ===
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=ragstudyemls;AccountKey=XXXXX;EndpointSuffix=core.windows.net
AZURE_BLOB_CONTAINER=eml-archive
# AZURE_BLOB_PREFIX=           # optional subfolder filter

# === Phase 2a: Qdrant Cloud ===
VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=https://xxxxxx.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
QDRANT_COLLECTION_NAME=email-rag
QDRANT_PREFER_GRPC=true         # gRPC is faster than REST for bulk upserts

# === Phase 2b: Local Qdrant in Docker (later) ===
# VECTOR_STORE_PROVIDER=qdrant
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=
# QDRANT_COLLECTION_NAME=email-rag
# QDRANT_PREFER_GRPC=true
```

To test **only Phase 1** (Azure loader + local vector store), omit the
Qdrant/Pinecone variables and keep `VECTOR_STORE_PROVIDER=simple` (or leave it
unset — `simple` is the default).

---

## Switching Back to Local-Only Mode

Set (or remove) the provider in `.env`:

```bash
VECTOR_STORE_PROVIDER=simple
```

Everything reverts to `SimpleVectorStore` on disk — no cloud vector store calls.

To switch from Qdrant Cloud to local Qdrant Docker later, keep
`VECTOR_STORE_PROVIDER=qdrant` and only change:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

---

## Cost Estimates

| Service | Tier | Monthly Cost |
|---|---|---|
| Azure Blob Storage (Cool, 48 GiB, LRS) | Pay-as-you-go | ~$0.48 |
| Qdrant Cloud | Free tier (varies by region) | ~$0.00 |
| LM Studio embeddings on local host | Local runtime | $0.00 |
| **Total ongoing** | | **~$0.48/month** |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ValueError: Azure connection string is required` | Set `AZURE_STORAGE_CONNECTION_STRING` in `.env` or as a codespace secret |
| `ValueError: QDRANT_URL environment variable is not set` | Set `QDRANT_URL` in `.env` when `VECTOR_STORE_PROVIDER=qdrant` |
| `Connection refused` to `host.docker.internal:1234` | Ensure LM Studio server is running on host and exposes an OpenAI-compatible endpoint |
| Qdrant insert/query auth errors | Verify `QDRANT_API_KEY` and `QDRANT_URL` match your cloud cluster |
| `ValueError: PINECONE_API_KEY environment variable is not set` | Set `PINECONE_API_KEY` in `.env` or as a codespace secret |
| `Connection string is either blank or malformed` | Make sure the full string is on one line with no line breaks |
| Pinecone upsert errors about dimension mismatch | Ensure the index was created with **1536** dimensions (matching `text-embedding-3-small`) |
| `ValueError: Metadata length ... is longer than chunk size` | Increase `RAG_CHUNK_SIZE` in `.env` (e.g. 2048). The metadata truncation in `models.py` should prevent this — run `pytest tests/test_document_metadata_limits.py` to verify |
| `PineconeApiException: Metadata size ... exceeds the limit of 40960 bytes` | Recipient/CC fields are too large. The `_MAX_RECIPIENT_FULL_LEN` cap in `models.py` should prevent this — run the metadata limit tests to verify |
| Exit code 139 (segfault / OOM kill) | Reduce `BATCH_SIZE` in `scripts/batch_index_to_vector_store.py` (default 200). Codespace smallest tier has ~2 GB RAM |
| Slow Azure downloads | Check the storage account region matches your runtime region |
| `scripts/batch_index_to_vector_store.py` stops mid-way | Just re-run — it resumes from `scripts/.vector_batch_checkpoint.txt` |
| Want to re-index Qdrant but keep schema | Run `poetry run python scripts/reset_qdrant_index.py` |
| Want to re-index Qdrant and delete schema too | Run `poetry run python scripts/reset_qdrant_index.py --drop-schema` |
| Want to re-index Pinecone from scratch | Run `poetry run python scripts/reset_pinecone_index.py` |
