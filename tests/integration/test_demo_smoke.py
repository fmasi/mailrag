import os
import pytest

pytestmark = pytest.mark.integration

@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1",
                    reason="integration: set RUN_INTEGRATION=1 (needs Qdrant+LLM+bge-m3)")
def test_build_and_query_10_enron():
    import main as m
    m.run_demo(num_samples=10, collection="mailrag-smoke",
               queries=["What is being scheduled?"])
    # success = no exception end-to-end (build + thread-aware query + answer)
