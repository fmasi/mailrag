# Email RAG System - Project Structure & Documentation

[![Test Suite](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml/badge.svg)](https://github.com/fmasi/mailrag/actions/workflows/test-suite.yml)

## 📁 Project Structure

```
mailrag/
├── .devcontainer/                # Dev container configuration
│   └── devcontainer.json
├── .github/                      # GitHub workflows and configuration
│   ├── dependabot.yml            # Dependency management
│   └── workflows/
│       └── test-suite.yml        # PR and main-branch test automation
├── .vscode/                      # VS Code configuration
│   ├── extensions.json
│   └── settings.json
├── docs/                         # Documentation
│   ├── README.md                 # This file
│   ├── QUICKSTART.md             # 5-minute setup guide
│   ├── ARCHITECTURE.md           # Design decisions & extensions
│   ├── EMAIL_PREPROCESSING.md    # ⭐ Reply-chain stripping & chunk size tuning guide
│   ├── INDEX.md                  # Navigation guide
│   ├── CLOUD_STORAGE_SETUP.md    # Azure Blob / Qdrant / Pinecone setup
│   ├── POETRY_MIGRATION.md       # Poetry dependency management notes
│   └── ARCHITECTURE_DIAGRAMS.py  # Visual system flow
├── src/                          # Main source code
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # ⭐ Configuration and LlamaIndex Settings
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # ⭐ Multi-source email loading API
│   │   ├── models.py            # ⭐ NormalizedEmail data structure
│   │   └── loaders/             # ⭐ Pluggable email source implementations
│   │       ├── __init__.py
│   │       ├── base.py          # Abstract EmailLoader interface
│   │       ├── enron.py         # Enron dataset (HuggingFace)
│   │       ├── mail_archive_x.py # Mail Archive X backups (.eml)
│   │       └── azure_blob.py    # Azure Blob Storage (.eml cloud)
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── indexer.py           # ⭐ Index creation and management
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── persist.py           # ⭐ Persistence (local, Pinecone cloud, or Qdrant)
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   └── engine.py            # ⭐ Query interface and RAG engine
│   │
│   └── __init__.py
│
├── scripts/                      # Utility scripts
│   ├── batch_index_to_vector_store.py # Batch index all emails to configured vector store
│   ├── analyze_email_lengths.py # ⭐ Profile email token lengths & choose chunk size
│   ├── debug_strip_reply_chain.py # ⭐ Diagnose reply-chain stripping on worst-case emails
│   ├── reset_pinecone_index.py  # Wipe Pinecone index & checkpoint
│   ├── reset_qdrant_index.py    # Reset Qdrant points or drop collection + checkpoint
│   ├── smoke_lmstudio_embedding.py # Live LM Studio embedding smoke check
│   └── validate_cloud_setup.py  # 8-check cloud deployment validation
│
├── tests/                        # Test files
│   ├── test_azure_blob_loader.py # Tests for Azure Blob loader
│   ├── test_document_metadata_limits.py # Tests for metadata size limits
│   ├── test_enron_loader.py     # Tests for Enron data loader
│   ├── test_load_emails.py      # Tests for email loading
│   ├── test_mail_archive_x_loader.py # Tests for Mail Archive X loader
│   ├── test_reply_chain_stripping.py # ⭐ Tests for reply-chain stripping & HTML extraction
│   ├── test_pinecone_storage.py # Tests for Pinecone vector store
│   ├── test_qdrant_storage.py   # Tests for Qdrant vector store
│   └── test_settings_embedding_provider.py # Tests for embedding provider config
│
├── storage/                      # Auto-created directory for persisted indexes
├── data_cache/                   # Auto-created directory for dataset cache
│
├── main.py                       # Example usage and entry point
├── examples_advanced.py          # Advanced features demo
├── pyproject.toml                # Poetry dependencies configuration
├── poetry.lock                   # Locked dependency versions
├── requirements.txt              # Legacy dependencies (for compatibility)
├── .env.example                  # Environment variables template
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore rules
```

## 🎯 Key Design Principles

## CI

- `.github/workflows/test-suite.yml` runs `python -m pytest tests/ -q` via Poetry on every pull request, on every push to `main`, and on manual dispatch.
- To make test success block merges into `main`, configure GitHub branch protection to require the `pytest` status check before merging.

### 1. **Modular Architecture**
Each module has a single responsibility:
- **config**: LLM and system configuration
- **data**: Multi-source email loading and processing
  - **loaders/**: Pluggable implementations (Enron, Mail Archive X, Azure Blob)
  - **models.py**: Unified data structures (NormalizedEmail)
- **indexing**: Index creation pipeline
- **storage**: Persistence and vector store management (local, Pinecone cloud, or Qdrant)
- **query**: Query interface and RAG execution

This separation makes it easy to swap components (e.g., different LLMs, vector stores, or email sources).

### 2. **Multi-Source Email Support**
The system can load emails from multiple sources through a pluggable loader architecture:
- **Enron Dataset**: Public email dataset from HuggingFace (100K+ emails)
- **Mail Archive X**: Local .eml file exports from personal email backups
- **Azure Blob Storage**: Cloud-hosted .eml files (~$0.48/mo for 68K emails on Cool tier)
- **Extensible**: Easy to add Gmail, Microsoft 365, or other sources

All sources produce a unified `NormalizedEmail` format, ensuring consistent behavior downstream.

### 3. **Metadata Extraction**
When loading emails, we extract:
- **sender**: Who sent the email
- **subject**: Email subject line
- **date**: When it was sent
- **source**: Where it came from (e.g., "enron", "mail_archive_x", "azure_blob")
- **recipients**: Email recipients (when available)

This metadata enables powerful queries like "emails from john@example.com about meetings".

### 4. **Efficiency & Persistence**
The first time you run the system:
- Loads emails from your chosen source (Enron, Mail Archive X, or Azure Blob)
- Embeds all emails (creates embeddings using your LLM)
- Saves everything to `./storage` (local), Pinecone, or Qdrant

Subsequent runs:
- Load the pre-computed embeddings from disk or your configured cloud vector store
- Skip the expensive embedding step
- Query in seconds instead of minutes

### 5. **Configuration as Code**
The `Settings` object (LlamaIndex v0.10+) centralizes all configuration:
- Change LLM: Just update `RAGConfig.LLM_PROVIDER`
- Change embeddings: Just update `RAGConfig.EMBEDDING_MODEL`
- All components automatically use the new settings

No need to pass objects around or edit multiple files.

## ⚙️ Setup Instructions

### 1. Install Dependencies

**Using Poetry (recommended):**

```bash
# Install Poetry if not already installed
pip install poetry

# Install dependencies
poetry install
```

**Using pip (legacy):**

```bash
pip install -r requirements.txt
```

> **Note**: This project now uses Poetry for dependency management. See `docs/POETRY_MIGRATION.md` for details.

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-openai-key           # needed for OpenAI LLM/embeddings only
# PERPLEXITY_API_KEY=pplx-your-perplexity-key  (optional)
RAG_EMBEDDING_PROVIDER=lmstudio             # openai or lmstudio
RAG_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
RAG_EMBEDDING_API_BASE=http://host.docker.internal:1234/v1
# RAG_EMBEDDING_API_KEY=                    # optional for local LM Studio
RAG_INDEX_BATCH_SIZE=200                    # optional indexing batch size
RAG_DOWNLOAD_WORKERS=8                      # optional parallel Azure downloads
RAG_EMBEDDING_BATCH_SIZE=512               # optional texts per embedding request (512 for M-series)
RAG_EMBEDDING_NUM_WORKERS=4                # match LM Studio's parallel requests setting
RAG_LLM_TEMPERATURE=0.7
RAG_CHUNK_SIZE=2048
RAG_CHUNK_OVERLAP=20

# Cloud storage (optional — see docs/CLOUD_STORAGE_SETUP.md)
# AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
# AZURE_BLOB_CONTAINER=eml-archive
# VECTOR_STORE_PROVIDER=qdrant               # simple / qdrant / pinecone
# QDRANT_URL=https://xxxxxx.us-east-1-0.aws.cloud.qdrant.io:6333
# QDRANT_API_KEY=...
# QDRANT_COLLECTION_NAME=email-rag
# QDRANT_PREFER_GRPC=true
# PINECONE_API_KEY=pcsk_XXXXXXXXXXXX         # optional legacy cloud provider
# PINECONE_INDEX_NAME=email-rag
```

### 3. Run the System

**With Poetry:**

```bash
poetry run python main.py
```

**Without Poetry:**

```bash
python main.py
```

**First run**: Will download the dataset and create the index (5-10 minutes)
**Subsequent runs**: Will load from disk (10-30 seconds)

## 📚 Usage Examples

### Basic RAG Query

```python
from src.config.settings import RAGConfig
from src.indexing.indexer import EmailIndexer
from src.query.engine import EmailQueryEngine

# Initialize
RAGConfig.initialize_settings()

# Build/load index
index = EmailIndexer.build_index()

# Query
query_engine = EmailQueryEngine(index)
response = query_engine.query("What are the main topics discussed?")
print(response)
```

### Pure Retrieval (No LLM)

Returns the most relevant emails without generating responses:

```python
results = query_engine.retrieval_query("meeting schedule", top_k=5)
query_engine.print_query_results(results)
```

### Load From Azure Blob Storage

```python
from src.data.loader import load_emails

# Requires AZURE_STORAGE_CONNECTION_STRING in .env
docs = load_emails(source="azure_blob", num_samples=50)
```

### Use Cloud Vector Store (Qdrant or Pinecone)

When `VECTOR_STORE_PROVIDER=qdrant` (or `pinecone`) is set in `.env`, `StorageManager` automatically uses that backend:

```python
from src.storage.persist import StorageManager

# Loads index from configured cloud backend (instead of local ./storage)
index = StorageManager.load_index()
```

### Metadata-Filtered Query

Search with filters:

```python
results = query_engine.query_with_metadata_filter(
    query_text="meetings",
    sender="john@example.com",
    top_k=5
)
```

### Check Index Status

```python
from src.indexing.indexer import EmailIndexer

EmailIndexer.print_index_info()
```

### Force Rebuild Index

```python
index = EmailIndexer.build_index(force_rebuild=True)
```

## 🔧 Customization Guide

### Change LLM Provider

Edit `src/config/settings.py`:

```python
RAGConfig.LLM_PROVIDER = "openai"  # or "perplexity"
RAGConfig.LLM_MODEL = "gpt-4"      # or your preferred model
```

Or set environment variables in `.env`:
```
RAG_LLM_PROVIDER=openai
RAG_LLM_MODEL=gpt-4
RAG_LLM_TEMPERATURE=0.7
```
### Change Vector Store

The default vector store is `SimpleVectorStore` (local JSON, no extra dependencies).

To use Qdrant Cloud:

```bash
# In .env
VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=https://xxxxxx.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=email-rag
QDRANT_PREFER_GRPC=true
```

To use Pinecone serverless:

```bash
# In .env
VECTOR_STORE_PROVIDER=pinecone
PINECONE_API_KEY=pcsk_XXXXXXXXXXXX
PINECONE_INDEX_NAME=email-rag
```

When `VECTOR_STORE_PROVIDER=qdrant` or `VECTOR_STORE_PROVIDER=pinecone`, `StorageManager` automatically pushes/pulls vectors to/from that cloud provider (`qdrant` is the recommended default; `pinecone` remains an optional legacy path). Set it back to `simple` (or remove it) to revert to local storage.

> **Full setup guide**: See [`docs/CLOUD_STORAGE_SETUP.md`](CLOUD_STORAGE_SETUP.md) for Azure account creation, Qdrant Cloud provisioning, optional Pinecone setup, and batch indexing.

For LM Studio indexing on an M-series Mac, recommended values:

```bash
RAG_INDEX_BATCH_SIZE=200
RAG_DOWNLOAD_WORKERS=8
RAG_EMBEDDING_BATCH_SIZE=512   # larger batches keep GPU utilisation high per request
RAG_EMBEDDING_NUM_WORKERS=1    # keep at 1 — see note below
```

Indexing uses an async `IngestionPipeline` for the Qdrant path. **Keep `RAG_EMBEDDING_NUM_WORKERS=1`**: values above 1 launch separate OS processes that independently flood LM Studio's request queue faster than the GPU drains it, causing the queue to balloon unboundedly and suppressing the terminal progress bar. With `embed_batch_size=512`, a single async worker saturates the GPU for the full duration of each inference pass.

If the process exits with code `139` (OOM/segfault), reduce `RAG_EMBEDDING_BATCH_SIZE` first (try `128`, then `64`), then `RAG_INDEX_BATCH_SIZE`.

### Add Custom Metadata

In `src/data/loader.py`, add new fields to the metadata dict:

```python
metadata = {
    "sender": sender,
    "subject": subject,
    "date": date_str,
    "cc": cc,          # Add new field
    "bcc": bcc,        # Add new field
    "source": "enron_qa",
}
```

### Add Custom Query Types

In `src/query/engine.py`, add new methods to `EmailQueryEngine`:

```python
def custom_query(self, query_text: str):
    """Your custom query logic here"""
    pass
```

## 📊 How the System Works

### Workflow: Index Creation (First Run)

```
Email Source (Enron / Mail Archive X / Azure Blob)
    ↓
load_emails(source="enron" | "mail_archive_x" | "azure_blob")
    ↓
Extract metadata (sender, subject, date)
    ↓
Create Document objects
    ↓
IngestionPipeline.arun() [Qdrant] / VectorStoreIndex.from_documents() [other]
    ├─ Chunks documents via SentenceSplitter
    ├─ Embeds chunks via Settings.embed_model (async, num_workers concurrent requests)
    ├─ Stores embeddings in SimpleVectorStore (local), Qdrant, or Pinecone
    └─ Saves to ./storage or configured cloud collection/index (persistent)
    ↓
Ready for queries!
```

### Workflow: Query Execution

```
User Question
    ↓
query_engine.query("What happened?")
    ↓
1. Retrieve: Find 5 most similar emails using embeddings
    ├─ Uses metadata for filtering
    └─ Returns relevant context
    ↓
2. Generate: Pass context + question to LLM
    ├─ LLM sees: "Context: [emails] Question: [user query]"
    └─ LLM generates answer based on context
    ↓
Response to User
```

## 🚀 Performance Tips

1. **Use smaller datasets for testing**: `EmailIndexer.build_index(num_samples=100)`

2. **Cache embeddings**: System automatically caches to `./storage`

3. **Profile and tune chunk size for your email source**: Before indexing,
   run the preprocessing analysis to find the right chunk size for your data.
   Reply chains inflate email length dramatically — stripping them first makes
   a large difference. See [EMAIL_PREPROCESSING.md](EMAIL_PREPROCESSING.md)
   for the full workflow.
   ```bash
   python scripts/analyze_email_lengths.py --sample 2000
   python scripts/debug_strip_reply_chain.py --sample 300 --show 5
   ```
   Then set in `.env`:
   ```
   RAG_CHUNK_SIZE=512
   RAG_CHUNK_OVERLAP=64
   ```

4. **Use cheaper embeddings** for development:
   ```python
   RAGConfig.EMBEDDING_MODEL = "text-embedding-3-small"  # Cheaper than large
   ```

5. **Batch queries**: Process multiple questions without reloading index

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Check that `.env` file exists in project root
- Verify the key is correct
- Ensure no spaces around the `=` sign

### "No index found"
- First run: This is normal. The system will create one.
- Check that `./storage` directory exists after creation

### Slow first run
- Embedding 100K documents takes time (5-10 minutes)
- This is normal and happens only once
- Subsequent runs load from cache (30 seconds)

### Memory issues with full dataset
- Use `num_samples` parameter to test with smaller dataset
- Or use `SimpleVectorStore` (it's less memory-intensive than some alternatives)

## 📖 Learning Resources

Each module is heavily commented with "why" explanations, not just "what" code does.

Key files to read for understanding:
1. `src/config/settings.py` - How LlamaIndex Settings work
2. `src/storage/persist.py` - How persistence works
3. `src/data/loader.py` - How metadata extraction works
4. `src/query/engine.py` - How RAG queries work

## 🎁 Next Steps

1. ✅ Run `python main.py` to test the basic setup
2. Modify queries in `main.py` to test different questions
3. Increase dataset size: Remove `num_samples` parameter
4. Experiment with different LLMs and embedding models
5. Add custom metadata fields
6. ☁️ Set up cloud storage: Follow [`docs/CLOUD_STORAGE_SETUP.md`](CLOUD_STORAGE_SETUP.md) for Azure Blob + Qdrant Cloud
7. Build a web interface using FastAPI or Streamlit

---

Built with ❤️ using LlamaIndex and the EnronQA dataset
