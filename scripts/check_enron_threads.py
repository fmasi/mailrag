"""
DECISION (recorded 2026-06-02):

DIAGNOSTIC FINDING:
  The Enron HF dataset (`MichaelR207/enron_qa_0922`) stores raw RFC 5322 email
  text in the `email` field. `normalize_enron_record` currently extracts only
  From / Subject / Date / body — it does NOT populate `message_id`,
  `in_reply_to`, or `references` on `NormalizedEmail`. Consequently, every call
  to `compute_thread_id` receives empty strings and every email gets thread_id
  `""`; no real threading occurs today.

  Raw-header analysis (300 samples) shows that the Enron dataset DOES carry
  Message-ID headers in virtually every email, but In-Reply-To / References are
  rare or absent. This means multi-email threads via RFC 5322 reply chains are
  essentially zero in a 300-sample window.

RECOMMENDATION FOR TASK 6 — option (b): subject-based grouping as fallback:
  1. Extend `normalize_enron_record` to extract Message-ID, In-Reply-To, and
     References from the raw `email` text (reuse `_extract_email_field`).
  2. In `compute_thread_id` (or `normalize_enron_record`), when all three RFC
     5322 fields are empty, fall back to a normalised subject-slug as the
     thread_id: strip "Re:", "Fwd:", "FW:", leading/trailing whitespace,
     lowercase, collapse internal whitespace to a single space.
  3. This gives every email a non-empty thread_id and creates plausible
     conversation groups (emails with the same subject line group together),
     making the small→big contextual demo illustrative even without full RFC
     5322 coverage.
  4. Verify with this script (after Task 6) that multi-email threads appear in
     300 samples using the subject-slug fallback.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
from collections import Counter

# Import only the pure-Python threading module (no llama_index dependency).
from src.data.threading import compute_thread_id


def _extract_field(email_text: str, field_name: str) -> str:
    """Extract a header field from raw email text."""
    lines = email_text.split("\n")
    for line in lines:
        if line.lower().startswith(field_name.lower() + ":"):
            return line[len(field_name) + 1 :].strip()
    return ""


def _subject_slug(subject: str) -> str:
    """Normalise a subject line to a stable grouping key (strip reply prefixes)."""
    s = subject.strip()
    # Strip common reply/forward prefixes repeatedly
    prefix_re = re.compile(r"^(?:re|fwd?|fw)\s*:\s*", re.IGNORECASE)
    while prefix_re.match(s):
        s = prefix_re.sub("", s).strip()
    return re.sub(r"\s+", " ", s).lower()


n = int(sys.argv[1]) if len(sys.argv) > 1 else 300

# Load the dataset directly to avoid the settings.py / llama_index import chain.
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: `datasets` package not found. Install it in this env.")
    sys.exit(1)

DATASET_NAME = os.getenv("RAG_DATASET_NAME", "MichaelR207/enron_qa_0922")
CACHE_DIR = os.getenv("RAG_DATA_CACHE_DIR", "./data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"Loading {DATASET_NAME} (first {n} records)...")
ds = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)
ds_sample = ds.select(range(min(n, len(ds))))
print(f"Loaded {len(ds_sample)} records.\n")

# --- Pass 1: RFC 5322 header analysis ---
has_msgid = 0
has_irt = 0
has_refs = 0
has_any_reply = 0

by_tid_rfc: dict = {}
by_tid_subject: dict = {}

for record in ds_sample:
    email_text = record.get("email", "")
    msg_id = _extract_field(email_text, "Message-ID")
    irt = _extract_field(email_text, "In-Reply-To")
    refs = _extract_field(email_text, "References")
    subject = _extract_field(email_text, "Subject")

    if msg_id:
        has_msgid += 1
    if irt:
        has_irt += 1
    if refs:
        has_refs += 1
    if irt or refs:
        has_any_reply += 1

    # Threading via RFC 5322 headers only
    tid_rfc = compute_thread_id(msg_id, irt, refs)
    by_tid_rfc[tid_rfc] = by_tid_rfc.get(tid_rfc, 0) + 1

    # Threading with subject-slug fallback
    if tid_rfc:
        tid_subj = tid_rfc
    else:
        slug = _subject_slug(subject)
        tid_subj = slug if slug else (msg_id or f"_idx_{len(by_tid_subject)}")
    by_tid_subject[tid_subj] = by_tid_subject.get(tid_subj, 0) + 1

print("--- Raw header presence ---")
print(f"emails_with_Message-ID:               {has_msgid}/{n}")
print(f"emails_with_In-Reply-To:              {has_irt}/{n}")
print(f"emails_with_References:               {has_refs}/{n}")
print(f"emails_with_reply_headers (irt|refs): {has_any_reply}/{n}")

# RFC 5322 thread stats
sizes_rfc = Counter(by_tid_rfc.values())
multi_rfc = sum(1 for v in by_tid_rfc.values() if v > 1)
print("\n--- Thread stats: RFC 5322 headers only ---")
print("thread-size -> #threads:", dict(sorted(sizes_rfc.items())))
print(f"threads={len(by_tid_rfc)} emails={n} multi_email_threads={multi_rfc}")

# Subject-slug fallback thread stats
sizes_subj = Counter(by_tid_subject.values())
multi_subj = sum(1 for v in by_tid_subject.values() if v > 1)
print("\n--- Thread stats: RFC 5322 + subject-slug fallback ---")
print("thread-size -> #threads:", dict(sorted(sizes_subj.items())))
print(f"threads={len(by_tid_subject)} emails={n} multi_email_threads={multi_subj}")

# Show a few example subject-grouped threads
print("\n--- Sample multi-email threads (subject-slug) ---")
shown = 0
for tid, count in sorted(by_tid_subject.items(), key=lambda x: -x[1]):
    if count > 1:
        print(f"  tid={tid!r:60s}  count={count}")
        shown += 1
        if shown >= 10:
            break
if shown == 0:
    print("  (none)")
