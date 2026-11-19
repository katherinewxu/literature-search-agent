import json
import unittest

from litsearch.data import fixture_store
from litsearch.environment import LiteratureSearchEnv
from litsearch.models import EpisodeSpec
from litsearch.retrieval import BM25Retriever
from litsearch.runner import run_episode
from litsearch.static_query import StaticQueryAgent


class Scripted:
    """Writes two queries, then includes the first paper of every screened batch."""

    model = "scripted"

    def __init__(self):
        self.calls = []

    def complete(self, messages, *, temperature, seed):
        user = messages[-1]["content"]
        self.calls.append(user)
        if "search queries" in user:
            return json.dumps({"queries": ["depression exercise", "physical activity mood"]})
        ids = [line.split()[1] for line in user.split("PAPERS:", 1)[1].splitlines() if line.startswith("- W")]
        return json.dumps({"include": ids[:1]})


class StaticQueryTests(unittest.TestCase):
    def _run(self, screen, budget=4):
        store = fixture_store()
        review = store.review("fixture_depression")
        spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, budget, "keyword")
        backend = Scripted()
        agent = StaticQueryAgent(backend, screen=screen)
        run = run_episode(LiteratureSearchEnv(store, BM25Retriever(store.papers.values()), spec), agent)
        return backend, agent, run

    def test_queries_written_once_and_used_round_robin(self):
        backend, agent, run = self._run(True)
        self.assertEqual(sum("search queries" in c for c in backend.calls), 1)
        queries = [ev["action"].get("query") for ev in run["trace"] if ev["action"]["kind"] == "search"]
        self.assertTrue(queries)
        self.assertEqual(set(queries), {"depression exercise", "physical activity mood"})
        self.assertLessEqual(len(queries), 4)

    def test_screening_includes_only_selected(self):
        _, agent, run = self._run(True)
        n_screens = agent.stats["turns"] - 1
        self.assertLessEqual(len(run["predictions"]), n_screens)

    def test_include_all_without_screening(self):
        backend, _, run = self._run(False)
        self.assertEqual(len(backend.calls), 1)  # only the query-writing call
        shown = {i for ev in run["trace"] for i in (ev.get("result_ids") or [])}
        self.assertTrue(set(run["predictions"]) >= shown - {None})


if __name__ == "__main__":
    unittest.main()
