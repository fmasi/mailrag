# Quick Start Guide

*[← docs index](INDEX.md) · [README](../README.md) · full setup in [`SETUP.md`](SETUP.md)*

> **The fastest path is `make demo`.** From a fresh clone:
> ```bash
> pip install -r requirements.txt   # includes FlagEmbedding (bge-m3)
> cp .env.example .env              # add an LLM key/endpoint for summaries + answers
> make demo                         # starts Qdrant, builds the contextual index, runs thread-aware queries
> ```
> This runs the **thread-aware contextual pipeline** (`main.py::run_demo`) over 100 Enron
> emails — the §13 stack from the [case study](../README.md#case-study-what-the-cleanup--retrieval-choices-actually-bought).
> See the root [`README.md`](../README.md) for what it does and [`SETUP.md`](SETUP.md) for the
> full local `.eml` pipeline and the test suite.
>
> The manual steps and code patterns below are kept as a reference for wiring the pieces
> together yourself. The live query API is `build_hybrid_searcher(...).search_threads(query)`
> (see [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md#how-to-call-it)); the retired
> `EmailIndexer` / `EmailQueryEngine` classes referenced in some older snippets no longer exist.

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

**Using Poetry (recommended):**
```bash
# Install Poetry if not already installed
pip install poetry

# Install project dependencies
poetry install
```

**Using pip (legacy):**
```bash
# From project root
pip install -r requirements.txt
```

> **Note**: This project now uses Poetry for dependency management. See `docs/POETRY_MIGRATION.md` for details.

### Step 1b: Set Up Noise Filter Rules

```bash
# Copy the template files (noise_rules.yaml and whitelist_domains.yaml are gitignored)
cp config/noise_rules.template.yaml config/noise_rules.yaml
cp config/whitelist_domains.template.yaml config/whitelist_domains.yaml
```

Then run the discovery tool to grow your rules from your own index:
```bash
python scripts/noise.py discover
```

### Step 2: Configure API Keys
```bash
# Copy template
cp .env.example .env

# Edit .env and add your model/vector settings.
# LLM key is only needed if you run RAG query generation with OpenAI.
# OPENAI_API_KEY=sk-...
# Optional tuning:
# RAG_LLM_TEMPERATURE=0.7
# RAG_CHUNK_SIZE=2048
# RAG_CHUNK_OVERLAP=20

# Embeddings provider (phase 1 default for local LM Studio):
# RAG_EMBEDDING_PROVIDER=lmstudio
# RAG_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
# RAG_EMBEDDING_API_BASE=http://host.docker.internal:1234/v1
# RAG_EMBEDDING_API_KEY=             # optional for local LM Studio

# Cloud storage (optional — see docs/CLOUD_STORAGE_SETUP.md):
# AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
# AZURE_BLOB_CONTAINER=eml-archive
# AZURE_BLOB_PREFIX=
# VECTOR_STORE_PROVIDER=qdrant           # or "simple"/"pinecone"
# QDRANT_URL=https://xxxxxx.us-east-1-0.aws.cloud.qdrant.io:6333
# QDRANT_API_KEY=...
# QDRANT_COLLECTION_NAME=email-rag
# QDRANT_PREFER_GRPC=true
```

### Step 3: Run the System

**With Poetry:**
```bash
# First time (creates index):
poetry run python main.py

# Subsequent times (loads from cache):
poetry run python main.py
```

**Without Poetry:**
```bash
# First time (creates index):
python main.py

# Subsequent times (loads from cache):
python main.py
```

That's it! 🎉

---

## 📚 Usage Patterns

### Pattern 1: Simple Retrieval (What emails match this?)
```python
from src.query.hybrid import build_hybrid_searcher

# Point at a built collection (see `make demo` / SETUP.md for how it's built)
searcher = build_hybrid_searcher("mailrag-demo", mode="hybrid")

# Plain retrieval — a list of NodeWithScore objects
nodes = searcher.search("meeting schedule")

# Thread-aware retrieval — a list of ThreadContext objects (match a unit, get its whole thread)
contexts = searcher.search_threads("meeting schedule")
```
> See [`RETRIEVAL_GUIDE.md` → How to call it](RETRIEVAL_GUIDE.md#how-to-call-it) for the full
> `HybridSearcher` API.

### Pattern 2: Load from Different Sources
```python
from src.data.loader import load_emails

# Load from Enron (default public dataset)
docs = load_emails(source="enron", num_samples=100)

# Load from Mail Archive X backup
docs = load_emails(
    source="mail_archive_x",
    backup_dir="/path/to/mail_archive_backup",
    num_samples=50
)

# Load from Azure Blob Storage (cloud .eml files)
# Requires AZURE_STORAGE_CONNECTION_STRING in .env
docs = load_emails(source="azure_blob", num_samples=50)
```

> **☁️ Cloud Storage**: To set up Azure Blob + Qdrant Cloud for production-scale
> email data, see [`docs/CLOUD_STORAGE_SETUP.md`](CLOUD_STORAGE_SETUP.md).

### Pattern 3: RAG Query (What's the answer based on my emails?)
```python
response = engine.query("What were the main decisions made in this email thread?")
print(response)
```

### Pattern 4: Filtered Search (Emails from someone about something)
```python
results = engine.query_with_metadata_filter(
    query_text="deadline",
    sender="john@acme.com",
    top_k=5
)
```

### Pattern 5: Batch Processing (Many questions)
```python
questions = [
    "What projects were discussed?",
    "Who proposed the budget changes?",
    "When is the next meeting?",
]

for q in questions:
    response = engine.query(q)
    print(f"Q: {q}\nA: {response}\n")
```

---

## 🔧 Key Configuration Options

### Use GPT-4 (higher quality, more expensive)
In `src/config/settings.py`:
```python
RAGConfig.LLM_MODEL = "gpt-4"
```

### Use smaller embeddings (cheaper)
In `src/config/settings.py`:
```python
RAGConfig.EMBEDDING_MODEL = "text-embedding-3-small"
```

### Test with a small dataset first
`run_demo` (in `main.py`) takes a sample count:
```python
from main import run_demo
run_demo(num_samples=50)   # 50 Enron emails instead of the default 100
```

### Force a rebuild of the collection
`build_contextual_index(..., recreate=True)` recreates the collection from scratch
(this is what `run_demo` does by default). Call it directly to rebuild a custom collection:
```python
from src.indexing.contextual_index import build_contextual_index
build_contextual_index(emails, collection="my-collection", embedder=embedder, recreate=True)
```

---

## 📂 Where Everything Lives

```
config/
└── noise_rules.yaml → Noise filter rules (sender domains, regex patterns) — edit to add new categories

src/
├── config/settings.py → Change LLM, embeddings, chunk size
├── data/
│   ├── loader.py → Source-agnostic email loading API
│   ├── models.py → NormalizedEmail data structure
│   ├── noise_filter.py → Rule-based noise classifier (pre- and post-index)
│   └── loaders/ → Pluggable email sources
│       ├── enron.py → Enron dataset (HuggingFace)
│       ├── mail_archive_x.py → Mail Archive X (.eml backups)
│       └── azure_blob.py → Azure Blob Storage (.eml cloud)
├── indexing/contextual_index.py → build_contextual_index(): clean → summary → embed → upsert
├── storage/persist.py → Handle disk, Pinecone, or Qdrant storage
└── query/hybrid.py → HybridSearcher: .search() and .search_threads()

scripts/ → Utility scripts
├── batch_index_to_vector_store.py → Batch-index emails; applies noise_rules.yaml before embedding
├── noise.py discover       → Interactive: classify unknown domains; prompts [y/n/w/2/3] per domain
│                             [y]=add rule  [n]=skip  [w]=whitelist  [2]=deep inspect  [3]=read email
│                             Use --auto to write rules without prompting (original behaviour)
├── noise.py purge          → Interactive: remove rule-matched noise from Qdrant and/or Azure Blob
├── noise.py deep-clean     → Per-email LLM cleanup for ambiguous domains (gmail, outlook …)
│                             Deletes confirmed noise from Qdrant + Azure Blob; extracts rules if possible
│   (combine: noise.py discover --auto --deep-clean  runs all three in one pass)
├── explore_topics.py → Analyse top topics in a sample of indexed emails
├── reset_pinecone_index.py → Wipe Pinecone index and checkpoint
├── reset_qdrant_index.py → Clear Qdrant points (default) or drop collection
├── smoke_lmstudio_embedding.py → Live LM Studio embedding smoke test
└── validate_cloud_setup.py → 8-check cloud deployment validation

tests/ → Test files for all modules
├── test_noise_filter.py → Tests for NoiseFilter (loading, matching, edge cases)
├── test_azure_blob_loader.py → Tests for Azure Blob loader
├── test_enron_loader.py → Tests for Enron data loader
├── test_load_emails.py → Tests for email loading
├── test_mail_archive_x_loader.py → Tests for Mail Archive X loader
├── test_pinecone_storage.py → Tests for Pinecone vector store
├── test_qdrant_storage.py → Tests for Qdrant vector store
└── test_document_metadata_limits.py → Tests for metadata size limits

main.py → Examples of using the system
examples_advanced.py → More complex use cases
playground.ipynb → Interactive playground notebook
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "OPENAI_API_KEY not found" | Add key only when using OpenAI LLM or OpenAI embeddings |
| "QDRANT_URL environment variable is not set" | Set `QDRANT_URL` when `VECTOR_STORE_PROVIDER=qdrant` |
| First run is slow | Normal! Embedding 100K documents takes time |
| "No index found" | First run hasn't completed. Be patient ~5-10 min |
| Out of memory | Use `num_samples` parameter, or increase chunk size |

---

## 📊 What Gets Saved Where

```
After first run:
./storage/                    ← Saved embeddings and index (local mode)
./data_cache/                 ← Downloaded dataset (Enron, cached)
                                 or path to Mail Archive X backup
./.env                        ← Your API keys (don't share!)

With VECTOR_STORE_PROVIDER=qdrant (recommended) or VECTOR_STORE_PROVIDER=pinecone (optional legacy):
    Vectors live in your configured cloud vector store (no local ./storage needed)
```

Subsequent runs load from `./storage` (local) or your configured cloud vector store and skip the expensive embedding step.

---

## 🎓 Learning Path

1. ✅ Run `make demo` (or `python main.py`) to see it work
2. Read the root [`README.md`](../README.md) for the overview and case study
3. Modify the example queries in `main.py::run_demo`
4. Read [`SETUP.md`](SETUP.md) to run the full local `.eml` pipeline
5. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) to understand the design
6. Read [`RETRIEVAL_GUIDE.md`](RETRIEVAL_GUIDE.md) and [`EXPERIMENTS.md`](EXPERIMENTS.md) for the retrieval stack and measured findings
7. Extend with your own custom queries

---

## 💡 Common Customizations

### Add custom metadata field
Edit `src/data/loader.py`, add to metadata dict:
```python
metadata = {
    "sender": sender,
    "subject": subject,
    "your_field": your_value,  # Add this line
}
```

### Add a custom query helper
Wrap `HybridSearcher` in `src/query/hybrid.py` (or a helper of your own):
```python
def find_urgent_threads(searcher):
    """Find threads about urgent matters."""
    return searcher.search_threads("urgent OR critical")
```

### Use Perplexity instead of OpenAI
In `src/config/settings.py`:
```python
RAGConfig.LLM_PROVIDER = "perplexity"
RAGConfig.LLM_MODEL = "sonar"
```

---

## 📈 Performance Tips

- **First run**: 5-10 minutes (embeds 100K emails)
- **Subsequent runs**: 30 seconds to load from cache
- **Per query**: 1-3 seconds (retrieval + LLM)
- **To speed up**: Use smaller chunks (`CHUNK_SIZE = 512`) or cheaper LLM
- **Env-based tuning**: Set `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, or `RAG_LLM_TEMPERATURE` in `.env`

---

## 🎯 Next Steps

1. ✅ Get it working with `python main.py`
2. Modify queries in `main.py` to test your questions
3. Increase dataset: Remove `num_samples` parameter
4. Add custom metadata extraction
5. ☁️ Set up cloud storage (Azure Blob + Qdrant Cloud) — see [`docs/CLOUD_STORAGE_SETUP.md`](CLOUD_STORAGE_SETUP.md)
6. Deploy with FastAPI or Streamlit

---

**Questions?** Check the docstrings in each Python file - they explain the "why" behind the code.
