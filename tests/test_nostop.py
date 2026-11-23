import json
import unittest

from litsearch.data import fixture_store
from litsearch.environment import LiteratureSearchEnv
from litsearch.llm import LLMSearchAgent
from litsearch.models import EpisodeSpec
from litsearch.nostop import NoStopAgent
from litsearch.retrieval import BM25Retriever
from litsearch.runner import run_episode


class Script:
    """Replies with the given action kinds in order (last one repeats)."""

    model = "scripted"

    def __init__(self, kinds):
        self.kinds, self.calls = list(kinds), []

    def complete(self, messages, *, temperature, seed):
        self.calls.append(messages[-1]["content"])
        kind = self.kinds[min(len(self.calls) - 1, len(self.kinds) - 1)]
        action = {"kind": kind, "query": "depression exercise"} if kind == "search" else {"kind": kind}
        return json.dumps({"include": [], "exclude": [], "action": action})


def run(kinds, wrap, budget=3):
    store = fixture_store()
    review = store.review("fixture_depression")
    spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, budget, "hybrid")
    backend = Script(kinds)
    agent = LLMSearchAgent(backend=backend, name="t")
    if wrap:
        agent = NoStopAgent(agent, max_refusals=2)
    out = run_episode(LiteratureSearchEnv(store, BM25Retriever(store.papers.values()), spec), agent)
    out["searches"] = sum(ev["action"]["kind"] == "search" for ev in out["trace"])
    return backend, agent, out


class NoStopTests(unittest.TestCase):
    # budget 3 = 1 for the seeds + 2 searches

    def test_unwrapped_agent_stops_immediately(self):
        backend, _, out = run(["stop"], wrap=False)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(out["searches"], 0)

    def test_rejected_stop_leads_to_search_and_nudge_in_prompt(self):
        backend, agent, out = run(["stop", "search", "search", "stop"], wrap=True)
        self.assertEqual(out["searches"], 2)
        self.assertEqual(out["metrics"]["charged_cost"], 3)  # budget spent
        self.assertIn("stop rejected", backend.calls[1])
        self.assertEqual((agent.rejected, agent.honoured_early), (1, 0))

    def test_persistent_stop_is_honoured_after_max_refusals(self):
        backend, agent, out = run(["stop"], wrap=True)
        self.assertEqual(len(backend.calls), 3)  # original + 2 re-asks
        self.assertEqual((agent.rejected, agent.honoured_early), (2, 1))
        self.assertEqual(out["searches"], 0)

    def test_stop_after_budget_spent_is_not_rejected(self):
        _, agent, out = run(["search", "search", "stop"], wrap=True)
        self.assertEqual(agent.rejected, 0)
        self.assertEqual(out["searches"], 2)


if __name__ == "__main__":
    unittest.main()
