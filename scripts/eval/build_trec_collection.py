import mailbox
import os
import sys

WT = "/Users/fmasi/Git/mailrag/.claude/worktrees/p2-backend-agnostic"
sys.path.insert(0, WT)
os.chdir(WT)
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from src.data.models import NormalizedEmail
from src.indexing.contextual_index import build_contextual_index
from src.ingest.embedder import BgeM3Embedder

MBOX = "/Users/fmasi/msgvault-eval-proof/trec/enron_trec.mbox"
emails = []
mb = mailbox.mbox(MBOX)
for msg in mb:
    mid = (msg.get("Message-ID") or "").strip().strip("<>")
    docid = mid.rsplit("@", 1)[0] if mid else None
    if not docid:
        continue
    try:
        body = (
            msg.get_content() if msg.get_content_maintype() == "text" else (msg.get_payload() or "")
        )
    except Exception:
        body = msg.get_payload() or ""
    if isinstance(body, list):
        body = " ".join(str(p) for p in body)
    subj = msg.get("Subject", "") or ""
    emails.append(
        NormalizedEmail(
            sender="trec@enron.com",
            subject=subj,
            date=None,
            body=(body or " "),
            source="trec",
            source_id=docid,
            recipients="r@enron.com",
            cc=None,
            message_id=docid,
        )
    )
print(f"parsed {len(emails)} TREC messages", flush=True)
res = build_contextual_index(
    emails,
    collection="trec-bge",
    embedder=BgeM3Embedder(),
    embed_summary=False,
    recreate=True,
    apply_noise_filter=False,
    qdrant_url=os.environ["QDRANT_URL"],
)
print(f"DONE trec-bge: kept={res.kept_emails} chunks={res.chunks}", flush=True)
