"""Generate a grounded answer from retrieved thread contexts.

Shared by the demo (main.run_demo), the onboard report, and `mailrag query` so
there is one prompt and one answer path.
"""
from src.llm.client import make_client, chat, default_model


def answer_from_threads(query, contexts, *, client=None, model=None, k=3):
    """Answer ``query`` using only the top-``k`` retrieved thread contexts.

    ``contexts`` is the list returned by ``HybridSearcher.search_threads`` — each
    item exposes a ``.text`` attribute. Returns a fallback string when nothing
    was retrieved."""
    if not contexts:
        return "No relevant threads retrieved."
    client = client or make_client()
    model = model or default_model()
    joined = "\n\n---\n\n".join(c.text for c in contexts[:k])
    prompt = (
        "Answer the question using only these email threads.\n\n"
        f"Threads:\n{joined}\n\nQuestion: {query}\nAnswer:"
    )
    return chat(client, model, prompt)
