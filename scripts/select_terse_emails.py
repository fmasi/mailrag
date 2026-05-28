# scripts/select_terse_emails.py
"""Surface candidate TERSE emails from a collection for the recall test.

Terse = short real body (from _node_content .text) AND a non-empty summary present.
Read-only Qdrant REST (stdlib). Prints candidates (ref_doc_id, subject, body, summary)
for manual curation — query drafting must use subject/thread, NOT the summary.

  python scripts/select_terse_emails.py --collection work-rag --max-body 150 --limit 12
"""
import argparse, json, urllib.request

def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def body_text(pl):
    raw = pl.get("_node_content")
    if not raw:
        return ""
    try:
        return (json.loads(raw).get("text") or "").strip()
    except (ValueError, TypeError):
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:6333")
    ap.add_argument("--collection", default="work-rag")
    ap.add_argument("--max-body", type=int, default=150)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    seen, candidates = set(), []
    offset = None
    while True:
        b = {"limit": 2000, "with_payload": True, "with_vector": False}
        if offset is not None:
            b["offset"] = offset
        res = post(args.base, f"/collections/{args.collection}/points/scroll", b)["result"]
        pts = res["points"]
        if not pts:
            break
        for p in pts:
            pl = p.get("payload") or {}
            rid = pl.get("ref_doc_id") or pl.get("document_id")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            body = body_text(pl)
            summary = (pl.get("summary") or "").strip()
            if 0 < len(body) <= args.max_body and summary:
                candidates.append((len(body), rid, pl.get("subject", ""), body, summary))
        offset = res.get("next_page_offset")
        if offset is None:
            break

    candidates.sort()  # shortest body first
    print(f"{len(candidates)} terse candidates (body <= {args.max_body} chars, summary present)\n")
    for n, (blen, rid, subj, body, summ) in enumerate(candidates[:args.limit], 1):
        print(f"--- candidate {n} | body={blen} chars | ref_doc_id={rid}")
        print(f"  subject: {subj[:90]}")
        print(f"  body   : {body[:160]!r}")
        print(f"  summary: {summ[:140]}")

if __name__ == "__main__":
    main()
