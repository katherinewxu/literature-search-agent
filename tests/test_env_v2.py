import json
import unittest

from litsearch.data import fixture_store
from litsearch.environment import LiteratureSearchEnv
from litsearch.llm import LLMSearchAgent
from litsearch.models import EpisodeSpec
from litsearch.retrieval import BM25Retriever
from litsearch.runner import run_episode
from litsearch.web import _seed_fallback_queries


class ExcludeThenExpand:
    """Excludes everything shown (reading one paper first), then expands a screened-out paper if offered."""

    model = "scripted"

    def __init__(self):
        self.prompts = []

    def complete(self, messages, *, temperature, seed):
        user = messages[-1]["content"]
        self.prompts.append(user)
        unscreened = user.split("UNSCREENED PAPERS", 1)[1]
        ids = [line.split()[1] for line in unscreened.splitlines() if line.startswith("- W")]
        action = {"kind": "read", "paper_id": ids[0]} if len(self.prompts) == 1 and ids else {"kind": "stop"}
        if "STILL EXPANDABLE" in user and len(self.prompts) < 3:
            block = user.split("STILL EXPANDABLE", 1)[1].split("RECENT ACTIONS", 1)[0]
            seen = [line.split()[1] for line in block.splitlines() if line.startswith("- W")]
            action = {"kind": "expand_citers", "paper_id": seen[0]}
        return json.dumps({"include": [], "exclude": ids, "action": action})


class EnvV2Tests(unittest.TestCase):
    def _run(self, show_seen):
        store = fixture_store()
        review = store.review("fixture_depression")
        spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, 10, "citation")
        backend = ExcludeThenExpand()
        run = run_episode(LiteratureSearchEnv(store, BM25Retriever(store.papers.values()), spec),
                          LLMSearchAgent(backend, show_seen=show_seen))
        return backend, run

    def test_default_prompt_unchanged(self):
        backend, run = self._run(0)
        self.assertTrue(all("STILL EXPANDABLE" not in p for p in backend.prompts))
        self.assertEqual(len(backend.prompts), 2)  # dead end after the read: nothing left to expand
        self.assertNotIn("expand_citers", [ev["action"]["kind"] for ev in run["trace"]])

    def test_show_seen_lists_excluded_papers_for_expansion(self):
        backend, run = self._run(20)
        self.assertIn("STILL EXPANDABLE", backend.prompts[1])
        kinds = [ev["action"]["kind"] for ev in run["trace"]]
        self.assertIn("expand_citers", kinds)

    def test_seed_fallback_queries_shorten_titles(self):
        qs = _seed_fallback_queries("Patients Retransitioning from Biosimilar TNF Inhibitor to the Originator: A Systematic Review")
        self.assertTrue(qs and all("Systematic" not in q for q in qs))
        self.assertLessEqual(len(qs[-1].split()), 5)


if __name__ == "__main__":
    unittest.main()
