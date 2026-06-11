import os, sys, glob, json, random
WT="/Users/fmasi/Git/mailrag/.claude/worktrees/p2-backend-agnostic"
sys.path.insert(0, WT); os.chdir(WT)
os.environ.setdefault("QDRANT_URL","http://localhost:6333")
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import pyarrow.ipc as ipc
from src.data.models import NormalizedEmail
from src.ingest.embedder import BgeM3Embedder
from src.indexing.contextual_index import build_contextual_index

N_CORPUS=2000; N_QUERIES=360
f=[x for x in glob.glob("/Users/fmasi/Git/mailrag/data_cache/MichaelR207___enron_qa_0922/**/*.arrow",recursive=True) if "test" in x][0]
try: t=ipc.open_file(f).read_all()
except Exception: t=ipc.open_stream(f).read_all()
rows=t.to_pylist()
# keep rows with a usable question + a body
def parse(txt, path):
    subj=""; sender="someone@enron.com"; body=txt
    for line in txt.split("\n")[:8]:
        if line.startswith("Subject:"): subj=line[8:].strip()
        elif line.startswith("Sender:"): sender=(line[7:].strip() or sender)
    if "=====" in txt:
        b=txt.split("=====",1)[1].strip()
        if len(b)>40: body=b
    return NormalizedEmail(sender=sender, subject=subj, date=None, body=body,
                           source="enron", source_id=path, recipients="recipients@enron.com",
                           cc=None, message_id=path)

usable=[r for r in rows if r.get("questions") and r["questions"] and len((r.get("email") or ""))>120]
random.seed(13); random.shuffle(usable)
corpus_rows=usable[:N_CORPUS]
emails=[parse(r["email"], r["path"]) for r in corpus_rows]
print(f"corpus emails: {len(emails)}", flush=True)

# queries: first N_QUERIES corpus rows -> question + gold path (present in corpus)
queries=[{"query": r["questions"][0], "answer_message_id": r["path"], "category":"enron"}
         for r in corpus_rows[:N_QUERIES]]
with open("/tmp/enron_qa_queries.jsonl","w") as o:
    for q in queries: o.write(json.dumps(q)+"\n")
print(f"wrote {len(queries)} queries -> /tmp/enron_qa_queries.jsonl", flush=True)

res=build_contextual_index(emails, collection="enron-qa-bge", embedder=BgeM3Embedder(),
                           embed_summary=False, recreate=True, apply_noise_filter=False,
                           qdrant_url=os.environ["QDRANT_URL"])
print(f"DONE built enron-qa-bge: kept={res.kept_emails} chunks={res.chunks}", flush=True)
