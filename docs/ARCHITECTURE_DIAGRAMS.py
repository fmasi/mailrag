"""
SYSTEM ARCHITECTURE DIAGRAM & DATA FLOW

This document provides visual representation of how the Email RAG system works.
"""

DIAGRAM_1_SYSTEM_ARCHITECTURE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    EMAIL RAG SYSTEM ARCHITECTURE                          ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                   │
│                     (main.py / examples_advanced.py)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Example Queries:                                                          │
│  • query("What are main topics?")                                          │
│  • retrieval_query("meeting schedule")                                     │
│  • query_with_metadata_filter("deadline", sender="john@acme.com")          │
└────────────────┬────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       src/query/engine.py                                  │
│                  (EmailQueryEngine)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Responsibilities:                                                         │
│  ✓ RAG queries (retrival + LLM generation)                                 │
│  ✓ Pure retrieval (similarity search only)                                 │
│  ✓ Metadata filtering                                                       │
│  ✓ Result formatting & pretty printing                                      │
└────────────────┬────────────────────────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Vector   │  │ Metadata │  │ Document │
│ Store    │  │ Store    │  │ Store    │
│ (Local   │  │          │  │          │
│ Disk)    │  │          │  │          │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │    src/storage/persist.py        │
    │   (StorageManager)               │
    ├──────────────────────────────────┤
    │ load_index() ────→ Fast load     │
    │ create_and_save_index() → Create │
    │ get_or_create_index() → Smart    │
    └──────────────────┬───────────────┘
                       │
        ┌──────────────┼──────────────┐
        │ (exists?)    │ (not exists?)│
        ▼              ▼
    ┌──────────┐  ┌─────────────────────┐
    │  LOAD    │  │  CREATE & SAVE      │
    │ (fast)   │  │ (expensive once)    │
    └─────┬────┘  └────────┬────────────┘
          │                │
          │                ▼
          │         ┌─────────────────────┐
          │         │ src/indexing/       │
          │         │ indexer.py          │
          │         │ (EmailIndexer)      │
          │         ├─────────────────────┤
          │         │ build_index()       │
          │         │ Orchestrates whole  │
          │         │ pipeline            │
          │         └────────┬────────────┘
          │                  │
          │                  ▼
          │         ┌─────────────────────┐
          │         │ src/data/           │
          │         │ loader.py           │
          │         ├─────────────────────┤
          │         │ load_emails()       │
          │         │ source="enron" or   │
          │         │ "mail_archive_x"    │
          │         └────────┬────────────┘
          │                  │
          │       ┌──────────┼───────────┐
          │       ▼                      ▼
          │  ┌──────────────┐   ┌─────────────────┐
          │  │ Enron        │   │ MailArchiveX    │
          │  │ Loader       │   │ Loader          │
          │  │ (HuggingFace)│   │ (.eml files)    │
          │  └──────┬───────┘   └────────┬────────┘
          │         └──────────┬──────────┘
          │                    │
          │                    ▼
          │         ┌─────────────────────┐
          │         │ NormalizedEmail     │
          │         │   objects           │
          │         │ (unified format)    │
          │         └────────┬────────────┘
          │                  │
          │                  ▼
          │         ┌─────────────────────┐
          │         │ Document objects    │
          │         │ (LlamaIndex)        │
          │         └────────┬────────────┘
          │                  │
          │                  ▼
          │       ┌───────────────────────┐
          │       │ VectorStoreIndex      │
          │       │ .from_documents()     │
          │       ├───────────────────────┤
          │       │ EMBEDDING STEP:       │
          │       │ Uses Settings.embed   │
          │       │ model to embed all    │
          │       │ documents             │
          │       │                       │
          │       │ ⏱️ SLOW (5-10 min)   │
          │       │ 💰 EXPENSIVE         │
          │       └────────┬──────────────┘
          │                │
          └────────────────┼──────────────┐
                           │              │
                           ▼              ▼
                    ┌────────────┐   ┌────────────┐
                    │ Return to  │   │  Save to   │
                    │ User       │   │  Storage   │
                    └────────────┘   └──────┬─────┘
                                            │
                                            ▼
                                    ┌────────────────┐
                                    │ Future runs:   │
                                    │ Load instantly │
                                    │ 30 seconds ✓   │
                                    └────────────────┘
"""

DIAGRAM_2_QUERY_FLOW = """
╔════════════════════════════════════════════════════════════════════════════╗
║                         QUERY EXECUTION FLOW                               ║
╚════════════════════════════════════════════════════════════════════════════╝

USER QUESTION: "What were the main decisions?"
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ RETRIEVAL PHASE (Semantic Search)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. Embed query:                                                            │
│    Text: "What were the main decisions?"                                   │
│         ──→ Settings.embed_model ──→ [0.23, -0.15, 0.89, ...]            │
│                                                                             │
│ 2. Find similar embeddings (in SimpleVectorStore):                        │
│    Query embedding ──→ Calculate similarity ──→ Return top 5              │
│                                                                             │
│ 3. Retrieve documents:                                                     │
│    • Email #1234: "The decision was made to..."  (similarity: 0.92)       │
│    • Email #1567: "We agreed on the following..." (similarity: 0.87)      │
│    • Email #1891: "The decision tree shows..."    (similarity: 0.85)      │
│    • Email #2103: "After discussion, we chose..." (similarity: 0.81)      │
│    • Email #2456: "The decision process was..."   (similarity: 0.78)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GENERATION PHASE (LLM-based Answer)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Construct Prompt:                                                          │
│                                                                             │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ SYSTEM: "You are an email assistant. Answer using the context below."  │ │
│ │                                                                        │ │
│ │ CONTEXT:                                                              │ │
│ │ Email 1:                                                              │ │
│ │ From: john@acme.com                                                  │ │
│ │ Subject: Board Decision                                              │ │
│ │ The decision was made to proceed with the merger. All stakeholders   │ │
│ │ agreed on the terms discussed in the previous meetings.              │ │
│ │                                                                        │ │
│ │ Email 2:                                                              │ │
│ │ From: sarah@company.com                                              │ │
│ │ Subject: RE: Board Decision                                          │ │
│ │ After discussion, the team agreed to implement the decision starting │ │
│ │ next quarter...                                                      │ │
│ │ [More emails...]                                                      │ │
│ │                                                                        │ │
│ │ QUESTION: "What were the main decisions?"                            │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ Send to LLM (Settings.llm = OpenAI GPT-3.5/4 or Perplexity):             │
│                                                                             │
│ LLM generates answer:                                                      │
│ "Based on the emails, the main decisions were:                            │
│  1. Proceeding with the merger                                            │
│  2. Agreement on all merger terms                                         │
│  3. Implementation starting next quarter                                  │
│  ..."                                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
ANSWER TO USER: "Based on the emails, the main decisions were..."
"""

DIAGRAM_3_DATA_LIFECYCLE = """
╔════════════════════════════════════════════════════════════════════════════╗
║                     DATA LIFECYCLE (First Run vs Later Runs)              ║
╚════════════════════════════════════════════════════════════════════════════╝

FIRST RUN (Expensive, happens once):
───────────────────────────────────────

SOURCE OPTIONS:
┌────────────────────────────┐        ┌─────────────────────────────┐
│ Hugging Face Datasets      │   OR   │ Mail Archive X Backups      │
│ MichaelR207/enron_qa_0922  │        │ Local .eml files            │
└──────────┬─────────────────┘        └─────────────┬───────────────┘
           │                                        │
           │ load_emails(source="enron")            │ load_emails(
           │                                        │   source="mail_archive_x",
           │                                        │   backup_dir="...")
           └────────────────┬───────────────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ EmailLoader      │
                   │ (base.py)        │
                   │ Abstract class   │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
      ┌──────────────┐            ┌──────────────────┐
      │ Enron        │            │ MailArchiveX     │
      │ Loader       │            │ Loader           │
      │ (enron.py)   │            │ (mail_archive_   │
      │              │            │  x.py)           │
      └──────┬───────┘            └────────┬─────────┘
             │                             │
             └─────────────┬───────────────┘
                           ▼
                  ┌─────────────────┐
                  │ NormalizedEmail │
                  │ objects         │
                  │ (models.py)     │
                  │                 │
                  │ • sender        │
                  │ • subject       │
                  │ • date          │
                  │ • body          │
                  │ • source        │
                  │ • recipients    │
                  └────────┬────────┘
                           │
                           ▼
                  LlamaIndex Document Objects
                  │
                  │ VectorStoreIndex.from_documents()
                  │
                  ├─ Embedding (EXPENSIVE!)
                  │  Each document:
                  │  text → Settings.embed_model → [embeddings vector]
                  │  ⏱️ 5-10 minutes for 100K docs
                  │  💰 Costs money each time!
                  │
                  └─ Create index structure
                     Ready for similarity search
                  │
                  ▼
          storage_context.persist()
                  │
                  ├─ default__vector_store.json ← Embeddings! (Most important)
                  ├─ default__docstore.json
                  └─ default__index_store.json
                  │
                  ▼
              ./storage/ directory


SUBSEQUENT RUNS (Fast, happens every time after first):
────────────────────────────────────────────────────────

                ./storage/ directory
                  │
                  │ StorageManager.load_index()
                  │
                  ├─ Load default__vector_store.json
                  │  (All embeddings pre-computed!)
                  │
                  ├─ Load default__docstore.json
                  │
                  └─ Load default__index_store.json
                     │
                     ▼
                 VectorStoreIndex (READY!)
                  │
                  │ ⏱️ 30 seconds
                  │ 💰 No embedding cost!
                  │ 🎉 100x faster!
                  │
                  ▼
          Ready for millions of queries


KEY INSIGHT:
────────────
The embedding step (from text to vector) happens ONCE and is cached.
Every query after that reuses the cached embeddings.
This is why the system is efficient!
"""

DIAGRAM_4_METADATA_FLOW = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    METADATA EXTRACTION & USAGE                             ║
║                (Works identically for all email sources)                   ║
╚════════════════════════════════════════════════════════════════════════════╝

RAW EMAIL FROM DATASET:
─────────────────────
From: john@acme.com
To: team@acme.com
Date: Mon, Feb 7, 2024 10:30:00 -0500
Subject: Important: Q1 Strategy Update

Greetings Team,

After careful consideration, we have decided to...
[Email body continues...]


EXTRACTION PROCESS:
────────────────────────────────────────

SOURCE: ENRON (HuggingFace)
┌────────────────────────────────────────────────────────┐
│ normalize_enron_record() in src/data/models.py         │
│  ├─ _extract_email_field(email_text, "From")           │
│  ├─ _extract_email_field(email_text, "Subject")        │
│  ├─ _extract_email_field(email_text, "Date")           │
│  └─ _extract_email_body(email_text)                    │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
            NormalizedEmail(
                sender="john@acme.com",
                subject="Important: Q1 Strategy Update",
                date=datetime(...),
                body="After careful consideration...",
                source="enron",
                source_id="enron_12345"
            )

SOURCE: MAIL ARCHIVE X (.eml files)
┌────────────────────────────────────────────────────────┐
│ MailArchiveXLoader._parse_eml_file()                   │
│  ├─ email.message_from_bytes() - parse .eml            │
│  ├─ msg.get("From")                                    │
│  ├─ msg.get("Subject")                                 │
│  ├─ parsedate_to_datetime(msg.get("Date"))             │
│  ├─ msg.get("To") → recipients                         │
│  └─ _extract_email_body_from_message(msg)              │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
            NormalizedEmail(
                sender="john@acme.com",
                subject="Important: Q1 Strategy Update",
                date=datetime(...),
                body="After careful consideration...",
                source="mail_archive_x",
                source_id="/path/to/email.eml",
                recipients="team@acme.com"
            )

                   │
       ┌───────────┴───────────┐
       │ IDENTICAL STRUCTURE!  │
       └───────────┬───────────┘
                   │
                   ▼

DOCUMENT CREATION (NormalizedEmail.to_document()):
──────────────────────────────────────────────────

Document(
    text="After careful consideration, we have decided to...",  # Searchable content
    metadata={
        "sender": "john@acme.com",          # ← For filtering
        "subject": "Important: Q1 Strategy", # ← For context
        "date": "2024-02-07T10:30:00",      # ← For sorting (ISO format)
        "source": "enron" or "mail_archive_x", # ← Source identifier
        "source_id": "enron_12345" or "/path/to/email.eml",
        "recipients": "team@acme.com"       # ← (if available)
    },
    doc_id="enron_12345" or "mail_archive_x_0"
)

KEY INSIGHT: Both sources produce IDENTICAL Document structure!
            → Indexing, storage, and query modules work seamlessly
            → No source-specific code needed downstream


USAGE IN QUERIES (src/query/engine.py):
───────────────────────────────────────

Query 1: Retrieval with metadata display
  results = engine.retrieval_query("strategy")
  →  Result shown with: From, Subject, Date, Relevance Score

Query 2: Filter by sender
  results = engine.query_with_metadata_filter(
      "strategy",
      sender="john@acme.com"
  )
  → Only emails from john@acme.com about strategy

Query 3: LLM generation with context
  response = engine.query("What's the strategy?")
  → LLM sees: email sender, subject, date + content
  → More informed answer!


BENEFITS:
─────────
✓ Users can filter by sender/subject/date
✓ LLM has context about email origin
✓ Search results are more informative
✓ Easy to add custom metadata fields
"""

print(__doc__)
print(DIAGRAM_1_SYSTEM_ARCHITECTURE)
print(DIAGRAM_2_QUERY_FLOW)
print(DIAGRAM_3_DATA_LIFECYCLE)
print(DIAGRAM_4_METADATA_FLOW)
