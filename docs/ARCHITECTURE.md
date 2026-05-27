"""
Architecture and Extension Guide for the Email RAG System.

This file explains the design decisions and provides guidance for extending
the system with custom functionality.
"""

"""
═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

The Email RAG system is built on these principles:

1. SEPARATION OF CONCERNS
   ├── config/      → LLM & system configuration
   ├── data/        → Multi-source email loading and preprocessing
   │   ├── loaders/ → Pluggable email source implementations
   │   │   ├── base.py           → Abstract EmailLoader interface
   │   │   ├── enron.py          → Enron dataset (HuggingFace)
   │   │   ├── mail_archive_x.py → Mail Archive X backups (.eml)
   │   │   └── azure_blob.py     → Azure Blob Storage (.eml cloud)
   │   ├── models.py        → NormalizedEmail data structure
   │   ├── noise_filter.py  → Rule-based noise classifier (pre- and post-index)
   │   └── loader.py        → Source-agnostic loading API
   ├── indexing/    → Index creation and pipeline management
   ├── storage/     → Persistence (local disk, Qdrant cloud, or Pinecone cloud)
   └── query/       → User interface and query execution

   WHY: Each module has one responsibility, making it easy to modify or swap.
   
   MULTI-SOURCE DESIGN: The loader abstraction allows switching between email
   sources (Enron, Mail Archive X, Gmail, etc.) without changing downstream code.

2. LAZY EXECUTION
   ├── First run: Load data → Embed → Create index → Save (local or cloud vector store)
   └── Subsequent runs: Load from disk or cloud vector store (skip embedding)

   WHY: Embedding 100K documents is expensive. Caching saves time and money.

3. METADATA EXTRACTION
   Documents include:
   ├── sender: Who sent the email
   ├── subject: Email topic
   ├── date: When sent
   └── source: Where it came from

   WHY: Metadata enables filtering and contextual retrieval.

4. LLAMA_INDEX SETTINGS (Global Configuration)
   ├── Settings.llm → Which LLM to use
   ├── Settings.embed_model → Which embeddings to use
   └── Settings.chunk_* → How to split documents

   WHY: Avoids passing objects everywhere. All components use global settings.


═══════════════════════════════════════════════════════════════════════════════
DETAILED FLOW: FROM DATA TO QUERY RESPONSE
═══════════════════════════════════════════════════════════════════════════════

INITIALIZATION (main.py):
   ↓
RAGConfig.initialize_settings()
   ├─ Loads environment variables
   ├─ Creates LLM instance (OpenAI or Perplexity)
   ├─ Creates Embeddings instance (OpenAI embeddings)
   ├─ Sets global Settings object
   └─ All future LlamaIndex components use these defaults

INDEX CREATION (first run):
   ↓
EmailIndexer.build_index()
   ├─ Calls StorageManager.index_exists() → False
   ├─ Calls load_emails(source="enron") from src.data.loader
   │  ├─ Creates EnronDatasetLoader (or MailArchiveXLoader / AzureBlobEmailLoader)
   │  ├─ Loader.load() returns List[NormalizedEmail]
   │  │  ├─ Enron: Downloads from Hugging Face (cached locally)
   │  │  ├─ Mail Archive X: Parses .eml files from backup directory
   │  │  ├─ Azure Blob: Downloads .eml blobs from Azure cloud storage
   │  │  └─ All extract metadata (sender, subject, date, recipients)
   │  │
   │  ├─ Converts NormalizedEmail → Document objects
   │  └─ All sources produce identical Document structure
   │
   ├─ Calls validate_documents() to check quality
   │
   ├─ Calls StorageManager.create_and_save_index(documents)
   │  ├─ Creates VectorStoreIndex.from_documents()
   │  │  ├─ Uses Settings.embed_model to embed each doc
   │  │  ├─ Stores embeddings in SimpleVectorStore (local), Qdrant, or Pinecone
   │  │  └─ Prepares vector store for searching
   │  │
   │  └─ Calls storage_context.persist(persist_dir="./storage")
   │     ├─ Saves embeddings to storage/default__vector_store.json
   │     ├─ Saves metadata
   │     └─ Future runs will load from here
   │
   └─ Returns ready-to-query VectorStoreIndex

INDEX LOADING (subsequent runs):
   ↓
EmailIndexer.build_index()
   ├─ Calls StorageManager.index_exists() → True
   ├─ Calls StorageManager.load_index()
   │  ├─ Creates StorageContext.from_defaults(persist_dir="./storage")
   │  ├─ Loads SimpleVectorStore from disk
   │  ├─ Calls load_index_from_storage()
   │  └─ Returns loaded index (much faster!)
   │
   └─ No embedding needed! Uses cached embeddings.

QUERY EXECUTION:
   ↓
EmailQueryEngine.query("What happened?")
   ├─ Calls index.as_query_engine(similarity_top_k=5)
   │
   ├─ RETRIEVAL PHASE:
   │  ├─ Embeds user query using Settings.embed_model
   │  ├─ Finds 5 most similar email embeddings
   │  ├─ Retrieves corresponding email documents
   │  └─ Uses metadata for context/filtering
   │
   ├─ GENERATION PHASE:
   │  ├─ Constructs prompt: "Context: [emails]\n\nQuestion: [query]"
   │  ├─ Sends to Settings.llm (OpenAI or Perplexity)
   │  └─ LLM generates answer based on context
   │
   └─ Returns response to user


═══════════════════════════════════════════════════════════════════════════════
NOISE FILTERING
═══════════════════════════════════════════════════════════════════════════════

Automated and social-media emails (LinkedIn notifications, digests, marketing)
add no business value and inflate the index.  The system handles this in two
complementary passes.

PASS 1 — Pre-index filtering (fast, free, automatic)
   config/noise_rules.yaml defines categories of noise by sender domain,
   sender regex, or subject regex.  batch_index_to_vector_store.py loads
   the rules at startup and silently skips matching emails before embedding.

   To add a new noise category: edit config/noise_rules.yaml, commit.
   No code change required.

PASS 2 — Post-index cleanup (interactive, on-demand)
   scripts/noise.py purge scans an existing Qdrant collection using the same
   rules, shows a per-category summary, then asks:
     1. Delete matching vectors from Qdrant?
     2. Delete source .eml files from Azure Blob Storage?

   Each question is answered independently; no destructive action happens
   without explicit confirmation.

AI-ASSISTED DISCOVERY (scripts/noise.py discover)
   Groups the indexed collection by sender domain and classifies each unknown
   domain with the LLM.  Runs in interactive mode by default; use --auto to
   write rules without prompting.

   For each candidate domain the LLM gives an initial verdict (NOISE / CLEAN)
   based on a sample of subject lines.  In interactive mode you are prompted:

     [y] Add noise rule    [n] Skip (this run)    [w] Whitelist (never re-propose)
     [2] Deep inspect      [3] Read an email

   [2] fetches full emails, runs the LLM per-email, and shows a noise
   percentage — informing your decision without forcing it.
   [3] shows one email body at a time so you can judge manually.
   [w] saves the domain to config/whitelist_domains.yaml so it is never
   proposed again (use this for trusted business partners like Globex.com).

   Two domain types are handled differently:

   Dedicated noise domains (e.g. newsletter.acme.com)
     The entire domain is noise — a sender_domains rule is generated.

   General-purpose domains (gmail.com, outlook.com, yahoo.com ...)
     Blocking the whole domain would filter real business emails.  The LLM
     is asked for narrow sender_patterns or subject_patterns instead.
     If no reliable pattern can be found, use deep-clean for that domain.

   Results are merged into config/noise_rules.yaml with unique keys — no
   duplicate keys are ever created.  The git diff is the review step; commit
   what you want to keep, delete what you don't.

   Auto mode (no prompts, original behaviour):
     python scripts/noise.py discover --auto
     python scripts/noise.py discover --auto --deep-clean  # also deep-clean ambiguous

DEEP CLEAN (scripts/noise.py deep-clean)
   For emails from general-purpose domains that cannot be expressed as a rule.
   Fetches each email from Qdrant, classifies it individually with the LLM
   (batched at 10 emails per call to minimise cost), then:
     1. Deletes confirmed noise vectors from Qdrant.
     2. Optionally deletes source .eml files from Azure Blob Storage.
     3. Attempts to extract a reusable rule from the noise batch and, if
        found, merges it into config/noise_rules.yaml for future pre-index runs.

   Cleaning has value even when no reusable rule can be extracted.

RULE FILE FORMAT (config/noise_rules.yaml)
   categories:
     category_name:
       description: Human-readable label
       sender_domains:          # substring-anchored to address (@domain or .domain)
         - linkedin.com
       sender_patterns:         # regex against full sender string (YAML single-quoted)
         - 'no.?reply@'
       subject_patterns:        # regex against subject (YAML single-quoted)
         - '^\[EXTERNAL\]'

WHY: Keeping noise out of the index directly improves retrieval precision and
reduces embedding cost on every future indexing run.


═══════════════════════════════════════════════════════════════════════════════
MULTI-SOURCE EMAIL LOADING
═══════════════════════════════════════════════════════════════════════════════

The system supports loading emails from multiple sources through a pluggable
loader architecture. All sources produce identical NormalizedEmail objects,
ensuring consistent behavior regardless of the data origin.

SUPPORTED SOURCES:
──────────────────

1. ENRON DATASET (HuggingFace)
   Source: enron
   Type: Public email dataset
   Storage: Downloaded and cached locally
   
   Usage:
      from src.data.loader import load_emails
      docs = load_emails(source="enron", num_samples=1000)

2. MAIL ARCHIVE X BACKUPS
   Source: mail_archive_x
   Type: Local .eml file exports
   Storage: User-provided backup directory
   
   Usage:
      from src.data.loader import load_emails
      docs = load_emails(
          source="mail_archive_x",
          backup_dir="/path/to/backup",
          num_samples=500
      )

3. AZURE BLOB STORAGE
   Source: azure_blob
   Type: Cloud-hosted .eml files
   Storage: Azure Blob Storage container (Cool tier, ~$0.48/mo for 68K emails)
   Config: AZURE_STORAGE_CONNECTION_STRING, AZURE_BLOB_CONTAINER, AZURE_BLOB_PREFIX
   
   Usage:
      from src.data.loader import load_emails
      docs = load_emails(source="azure_blob", num_samples=500)

ARCHITECTURE OVERVIEW:
──────────────────────

┌──────────────────────────────────────────────────────────┐
│                  src/data/loader.py                      │
│              load_emails(source="...")                   │
└─────────────────┬────────────────────────────────────────┘
                  │
       ┌──────────┼──────────────────────┐
       ▼                     ▼           ▼
┌─────────────────┐   ┌──────────────┐   ┌──────────────┐
│ EnronDataset    │   │ MailArchiveX │   │ AzureBlob    │
│ Loader          │   │ Loader       │   │ EmailLoader  │
│ (enron.py)      │   │ (mail_       │   │ (azure_      │
│                 │   │ archive_x.py)│   │ blob.py)     │
└────────┬────────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             ▼
              ┌─────────────────┐
              │ NormalizedEmail │
              │   (models.py)   │
              ├─────────────────┤
              │ • sender        │
              │ • subject       │
              │ • date          │
              │ • body          │
              │ • source        │
              │ • recipients    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Document objects│
              │ (LlamaIndex)    │
              └─────────────────┘

WHY THIS DESIGN:
────────────────
✓ Adding new sources requires only one new loader class
✓ All sources produce identical output → downstream code unchanged
✓ Easy to test sources independently
✓ Backward compatible (load_enron_dataset() still works)

ADDING A NEW SOURCE:
────────────────────
1. Create new class inheriting from EmailLoader
2. Implement load() → returns List[NormalizedEmail]
3. Implement get_source_info() → returns Dict with metadata
4. Add to load_emails() function in loader.py
5. No changes needed to indexing, storage, or query modules!


═══════════════════════════════════════════════════════════════════════════════
HOW TO EXTEND: COMMON USE CASES
═══════════════════════════════════════════════════════════════════════════════

USE CASE 1: SWAP THE LLM
────────────────────────
Goal: Use GPT-4 instead of GPT-3.5, or use Perplexity

SOLUTION:
   Edit src/config/settings.py:
   
   RAGConfig.LLM_PROVIDER = "openai"
   RAGConfig.LLM_MODEL = "gpt-4"  # Higher quality, more expensive
   
   OR
   
   RAGConfig.LLM_PROVIDER = "perplexity"
   RAGConfig.LLM_MODEL = "pplx-7b-online"  # Cheaper, fast
   
   That's it! All components will use the new model.


USE CASE 2: ADD NEW METADATA FIELDS
─────────────────────────────────────
Goal: Extract CC, BCC, or attachments from emails

SOLUTION:
   Edit the appropriate loader in src/data/loaders/:
   
   For Enron emails (src/data/loaders/enron.py):
   Modify normalize_enron_record() in src/data/models.py:
   
   OLD:
      return NormalizedEmail(
          sender=sender or "unknown",
          subject=subject or "(no subject)",
          date=date_obj,
          body=body or email_text,
          source="enron",
          source_id=f"enron_{index}",
      )
   
   NEW:
      cc = _extract_email_field(email_text, "Cc")
      bcc = _extract_email_field(email_text, "Bcc")
      
      return NormalizedEmail(
          sender=sender or "unknown",
          subject=subject or "(no subject)",
          date=date_obj,
          body=body or email_text,
          source="enron",
          source_id=f"enron_{index}",
          recipients=cc,  # Use the existing recipients field
          # Add custom metadata in to_document() method if needed
      )
   
   For Mail Archive X (src/data/loaders/mail_archive_x.py):
   The loader already extracts recipients from the "To:" field!


USE CASE 3: ADD A NEW EMAIL SOURCE
───────────────────────────────────
Goal: Load emails from Gmail, Microsoft 365, or another source

SOLUTION:
   1. Create src/data/loaders/gmail.py:
   
      from src.data.loaders.base import EmailLoader
      from src.data.models import NormalizedEmail
      
      class GmailLoader(EmailLoader):
          def __init__(self, credentials_path: str):
              self.credentials = credentials_path
          
          def load(self, num_samples: Optional[int] = None) -> List[NormalizedEmail]:
              # Use Gmail API to fetch emails
              # Parse each email into NormalizedEmail
              emails = []
              for msg in gmail_messages:
                  email = NormalizedEmail(
                      sender=msg['from'],
                      subject=msg['subject'],
                      date=msg['date'],
                      body=msg['body'],
                      source="gmail",
                      source_id=msg['id'],
                      recipients=msg['to'],
                  )
                  emails.append(email)
              return emails
          
          def get_source_info(self) -> Dict[str, Any]:
              return {"source": "gmail", "credentials": self.credentials}
   
   2. Update src/data/loaders/__init__.py:
      
      from src.data.loaders.gmail import GmailLoader
      __all__ = [..., "GmailLoader"]
   
   3. Update src/data/loader.py, in load_emails():
      
      elif source == "gmail":
          if not backup_dir:  # or credentials_path
              raise ValueError("credentials required for gmail source")
          loader = GmailLoader(backup_dir)
   
   That's it! No changes needed to indexing, storage, or query modules.


USE CASE 4: ADD CUSTOM QUERY TYPES
───────────────────────────────────
───────────────────────────────────
Goal: Add specific query methods for your domain

SOLUTION:
   Add new methods to EmailQueryEngine in src/query/engine.py:
   
   def find_urgent_emails(self) -> List[dict]:
       """Find emails marked as urgent."""
       results = self.retrieval_query("urgent OR critical OR important")
       return results
   
   def find_emails_from_date_range(self, start_date, end_date) -> List[dict]:
       """Find emails within a date range."""
       # Use the retriever APIs to filter by date
       pass
   
   def find_decision_emails(self) -> List[dict]:
       """Find emails about decisions made."""
       results = self.retrieval_query("decided OR approved OR agreed")
       return results


USE CASE 5: SWITCH TO CLOUD VECTOR STORE
────────────────────────────────────────────
Goal: Use Qdrant cloud (or Pinecone) instead of local SimpleVectorStore

SOLUTION:
   Cloud vector stores are built-in. Set environment variables:

   In .env (Qdrant primary):
      VECTOR_STORE_PROVIDER=qdrant
      QDRANT_URL=https://...:6333
      QDRANT_API_KEY=...
      QDRANT_COLLECTION_NAME=email-rag

   Optional Pinecone legacy path:
      VECTOR_STORE_PROVIDER=pinecone
      PINECONE_API_KEY=pcsk_...
      PINECONE_INDEX_NAME=email-rag

   StorageManager automatically selects the provider from
   VECTOR_STORE_PROVIDER. Set it to "simple" (or omit)
   to keep using local SimpleVectorStore.
   
   For full setup details, see docs/CLOUD_STORAGE_SETUP.md.


USE CASE 6: ADD FILTERING TO QUERIES
──────────────────────────────────────
Goal: Only search within emails from specific years

SOLUTION:
   Add new query method or extend existing in src/query/engine.py:
   
   def query_by_year(self, query_text: str, year: int) -> List[dict]:
       """Query emails from a specific year."""
       results = self.retrieval_query(query_text, top_k=10)
       
       # Filter by year from metadata
       filtered = []
       for result in results:
           date_str = result['metadata'].get('date', '')
           if str(year) in date_str:
               filtered.append(result)
       
       return filtered[:5]


USE CASE 7: BATCH PROCESSING
──────────────────────────────
Goal: Process multiple datasets and queries efficiently

SOLUTION:
   Use the modular structure:
   
   def process_multiple_queries(queries: List[str]):
       # Load once
       index = EmailIndexer.build_index()
       query_engine = EmailQueryEngine(index)
       
       # Query many times (no reloading!)
       results = []
       for query in queries:
           response = query_engine.query(query)
           results.append(response)
       
       return results


═══════════════════════════════════════════════════════════════════════════════
PERFORMANCE OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════════

1. CHUNK SIZE TUNING
   ─────────────────
   Smaller chunks → More granular search, but more documents to store
   Larger chunks  → Faster queries, but less specific results
   
   Adjust in src/config/settings.py:
      RAGConfig.CHUNK_SIZE = 1024  # Try 512 for finer-grained search
      RAGConfig.CHUNK_OVERLAP = 20

   Or set in environment (.env or OS env):
      RAG_CHUNK_SIZE=1024
      RAG_CHUNK_OVERLAP=20

2. TOP-K TUNING
   ────────────
   More results → More context for LLM, but slower
   Fewer results → Faster, but less comprehensive
   
   Adjust in src/query/engine.py:
      index.as_query_engine(similarity_top_k=5)  # Try 3 for speed, 10 for quality

3. EMBEDDING MODEL CHOICE
   ──────────────────────
   Small (faster):  "text-embedding-3-small"
   Large (better):   "text-embedding-3-large"
   
   Choose in src/config/settings.py:
      RAGConfig.EMBEDDING_MODEL = "text-embedding-3-small"  # Cheaper, 75% cost

4. BATCH PROCESSING
   ────────────────
   Load index once, query many times (not per-query)
   
   In src/query/engine.py, I've shown how metadata filters
   can be applied without reloading the index.


═══════════════════════════════════════════════════════════════════════════════
DEBUGGING & MONITORING
═══════════════════════════════════════════════════════════════════════════════

GET INDEX INFO:
   EmailIndexer.print_index_info()
   → Shows storage size, document count, location

PURE RETRIEVAL (NO LLM):
   results = query_engine.retrieval_query("query")
   → See what documents are actually being retrieved
   → Useful for debugging why certain results appear

INSPECT METADATA:
   results = query_engine.retrieval_query("query")
   for result in results:
       print(result['metadata'])
   → See what metadata is being extracted

TRACE LLM CALLS (if using OpenAI):
   Set environment: OPENAI_LOG=debug_requests.log
   → See exactly what's being sent to the LLM


═══════════════════════════════════════════════════════════════════════════════
TESTING YOUR EXTENSIONS
═══════════════════════════════════════════════════════════════════════════════

1. Test with small dataset:
   index = EmailIndexer.build_index(num_samples=10)
   → Quick feedback without waiting for full index

2. Test in isolation:
   from src.data.loader import load_emails, validate_documents
   
   # Test Enron loader
   docs = load_emails(source="enron", num_samples=5)
   validate_documents(docs)
   
   # Test Mail Archive X loader (if you have backup)
   docs = load_emails(source="mail_archive_x", 
                      backup_dir="/path/to/backup", 
                      num_samples=5)
   validate_documents(docs)
   
   → Test data loading without creating index

3. Test queries:
   query_engine = EmailQueryEngine(index)
   results = query_engine.retrieval_query("test")
   print(results)  # See actual results, not LLM interpretation

4. Monitor changes:
   info_before = EmailIndexer.get_index_info()
   # Make changes
   info_after = EmailIndexer.get_index_info()
   # Compare


═══════════════════════════════════════════════════════════════════════════════
COMMON ISSUES & SOLUTIONS
═══════════════════════════════════════════════════════════════════════════════

ISSUE: "OPENAI_API_KEY not found"
SOLUTION: Add to .env file in project root

ISSUE: "No index found"
SOLUTION: First run takes 5-10 minutes. Be patient or test with num_samples=10

ISSUE: Queries are too slow
SOLUTION: Reduce similarity_top_k, test with smaller chunk_size, or use cheaper LLM

ISSUE: Metadata not extracted correctly
SOLUTION: Check email format in load_enron_dataset(), adjust parsing logic

ISSUE: Want to use different dataset
SOLUTION: Create a new loader class in src/data/loaders/ inheriting from EmailLoader,
          or modify load_emails() in src/data/loader.py to add your source


═══════════════════════════════════════════════════════════════════════════════
"""

# This file is documentation. See examples_advanced.py for practical examples.
