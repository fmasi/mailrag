"""
Topic exploration script for the indexed email collection.

Scrolls up to 1000 points directly from Qdrant (no embeddings needed),
extracts email subjects and senders, then asks the LLM to identify the
top recurring topics from the dataset.

Usage:
    python scripts/explore_topics.py
    python scripts/explore_topics.py --sample 500 --top 10
    python scripts/explore_topics.py --no-llm   # fast mode: just print subject stats
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_points(collection: str, limit: int, url: str, api_key: str | None):
    """Scroll the Qdrant collection and return up to `limit` payloads."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, api_key=api_key or None)

    payloads = []
    offset = None

    while len(payloads) < limit:
        batch_size = min(256, limit - len(payloads))
        results, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,   # no need to pull vectors for metadata analysis
        )
        for point in results:
            payloads.append(point.payload or {})

        if next_offset is None:
            break
        offset = next_offset

    return payloads


def strip_re_fwd(subject: str) -> str:
    """Remove Re:/Fwd:/FW: prefixes to normalise subjects."""
    return re.sub(r'^(re|fwd?|fw)\s*:\s*', '', subject.strip(), flags=re.IGNORECASE).strip()


def keyword_counts(subjects: list[str], top_n: int) -> list[tuple[str, int]]:
    """Return the most common individual words from subjects (excluding stopwords)."""
    stopwords = {
        "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "be",
        "on", "at", "with", "from", "by", "as", "your", "our", "my", "this",
        "that", "are", "was", "it", "we", "you", "i", "not", "re", "fw", "fwd",
        "no", "new", "up", "have", "", "about", "has", "can", "will", "please",
        "hi", "hello", "dear", "hey", "thank", "thanks",
    }
    words = []
    for s in subjects:
        words.extend(
            w for w in re.findall(r"[a-z']+", s.lower()) if w not in stopwords and len(w) > 2
        )
    return Counter(words).most_common(top_n)


def ask_llm_for_topics(subjects: list[str], top_kw: list[tuple[str, int]], top_n: int) -> str:
    """
    Send a representative sample of subjects + keyword stats to the LLM and ask
    it to identify the top recurring topics.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "(OPENAI_API_KEY not set — skipping LLM synthesis)"

    client = OpenAI(api_key=api_key)

    # Include up to 300 subjects so the prompt stays reasonably sized
    sample = subjects[:300]
    kw_lines = "\n".join(f"  {w}: {c}" for w, c in top_kw)
    subject_block = "\n".join(f"- {s}" for s in sample)

    prompt = f"""You are an analyst reviewing a sample of {len(subjects)} email subjects from a mailbox.

Top keyword frequencies (word: count):
{kw_lines}

A sample of the actual subjects:
{subject_block}

Based on this data, list the top {top_n} recurring topics or themes in this mailbox.
For each topic, give it a short label and a 1-sentence description of what kinds of emails fall under it.
Format your answer as a numbered list."""

    model = os.getenv("RAG_LLM_MODEL", "gpt-3.5-turbo").strip() or "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Explore top topics in indexed emails")
    parser.add_argument("--sample", type=int, default=1000, help="Number of Qdrant points to fetch (default: 1000)")
    parser.add_argument("--top", type=int, default=10, help="Number of top topics to report (default: 10)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM synthesis, just show keyword stats")
    args = parser.parse_args()

    qdrant_url = os.getenv("QDRANT_URL", "http://host.docker.internal:6333").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    collection = os.getenv("QDRANT_COLLECTION_NAME", "email-rag").strip()

    print(f"\n{'=' * 60}")
    print(f"  Email Topic Explorer")
    print(f"{'=' * 60}")
    print(f"  Collection : {collection}")
    print(f"  Qdrant URL : {qdrant_url}")
    print(f"  Sample size: {args.sample}")
    print(f"{'=' * 60}\n")

    # --- Fetch payloads ---
    print(f"Fetching up to {args.sample} points from Qdrant...")
    payloads = fetch_points(collection, args.sample, qdrant_url, qdrant_api_key)
    print(f"  Retrieved {len(payloads)} points.\n")

    if not payloads:
        print("No data found. Is the collection populated?")
        return

    # --- Extract metadata ---
    subjects_raw = [p.get("subject") or p.get("_node_content") or "" for p in payloads]
    subjects_raw = [s for s in subjects_raw if s]  # drop blanks

    # Some payloads store metadata nested differently — peek at a sample
    if not subjects_raw:
        print("Note: 'subject' key not found in payloads. Checking for nested metadata...")
        # LlamaIndex stores metadata in _node_content (JSON string) or metadata dict
        import json
        for p in payloads[:5]:
            nc = p.get("_node_content", "")
            if nc:
                try:
                    nc_data = json.loads(nc)
                    meta = nc_data.get("metadata", {})
                    print(f"  Sample metadata keys: {list(meta.keys())}")
                    break
                except Exception:
                    pass
        # Try extracting via _node_content JSON
        for p in payloads:
            nc = p.get("_node_content", "")
            if nc:
                try:
                    nc_data = json.loads(nc)
                    subj = nc_data.get("metadata", {}).get("subject", "")
                    if subj:
                        subjects_raw.append(subj)
                except Exception:
                    pass

    senders = [p.get("sender", "") for p in payloads]

    # Normalise: strip Re:/Fwd:
    subjects_clean = [strip_re_fwd(s) for s in subjects_raw if s.strip()]
    subjects_clean = [s for s in subjects_clean if s]

    print(f"  Subjects found       : {len(subjects_clean)}")
    print(f"  Unique subjects      : {len(set(subjects_clean))}")
    unique_senders = len(set(s for s in senders if s))
    if unique_senders:
        print(f"  Unique senders       : {unique_senders}")

    # --- Subject frequency ---
    print(f"\n--- Top {args.top} most common subjects ---")
    subject_counts = Counter(subjects_clean).most_common(args.top)
    for i, (subj, cnt) in enumerate(subject_counts, 1):
        bar = "#" * min(cnt, 40)
        print(f"  {i:2}. ({cnt:4}x) {bar}")
        print(f"        {subj[:80]}")

    # --- Keyword frequency ---
    top_kw = keyword_counts(subjects_clean, top_n=30)
    print(f"\n--- Top keywords in subjects ---")
    for word, count in top_kw[:20]:
        bar = "#" * min(count // max(1, len(subjects_clean) // 50), 40)
        print(f"  {word:<20} {count:4}  {bar}")

    # --- Sender stats ---
    if unique_senders:
        sender_counts = Counter(s for s in senders if s).most_common(10)
        print(f"\n--- Top {min(10, len(sender_counts))} senders ---")
        for sender, cnt in sender_counts:
            print(f"  {cnt:4}x  {sender[:60]}")

    # --- LLM topic synthesis ---
    if args.no_llm:
        print("\n(LLM synthesis skipped via --no-llm)")
        return

    print(f"\n{'=' * 60}")
    print(f"  Asking LLM to identify top {args.top} topics...")
    print(f"{'=' * 60}\n")

    result = ask_llm_for_topics(subjects_clean, top_kw, args.top)
    print(result)
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
