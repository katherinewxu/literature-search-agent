import unittest

from litsearch.data import fixture_store
from litsearch.environment import LiteratureSearchEnv
from litsearch.models import Action, ActionKind, EpisodeSpec
from litsearch.retrieval import BM25Retriever


def make_env(mode="hybrid", budget=20):
    store = fixture_store()
    review = store.review("fixture_depression")
    return LiteratureSearchEnv(
        store,
        BM25Retriever(store.papers.values()),
        EpisodeSpec(review.id, review.question, review.eligibility_criteria, budget, mode),
    )


class EnvironmentTests(unittest.TestCase):
    def test_leakage_filters(self):
        env = make_env()
        self.assertNotIn("W90", env.allowed_ids)
        self.assertNotIn("W91", env.allowed_ids)
        observation = env.observe().to_dict()
        self.assertNotIn("gold", str(observation).lower())

    def test_mode_restrictions(self):
        env = make_env("citation")
        result = env.step(Action(ActionKind.SEARCH, query="relapse", k=10))
        self.assertEqual(result.error.code, "tool_unavailable")
        self.assertEqual(result.charged_cost, 0)

        env = make_env("keyword")
        paper_id = env.seed_ids[0]
        result = env.step(Action(ActionKind.EXPAND_REFS, paper_id=paper_id, k=10))
        self.assertEqual(result.error.code, "tool_unavailable")

    def test_costs_and_free_decisions(self):
        env = make_env(budget=10)
        paper_id = env.seed_ids[0]
        read = env.step(Action(ActionKind.READ, paper_id=paper_id))
        self.assertEqual(read.charged_cost, 1)
        self.assertEqual(read.remaining_budget, 8)
        include = env.step(Action(ActionKind.INCLUDE, paper_id=paper_id))
        self.assertEqual(include.charged_cost, 0)
        self.assertIn(paper_id, env.included)

    def test_paper_must_be_exposed(self):
        env = make_env()
        result = env.step(Action(ActionKind.INCLUDE, paper_id="W7"))
        if "W7" in env.seed_ids:
            self.skipTest("fixture rank exposed W7")
        self.assertEqual(result.error.code, "not_exposed")

    def test_search_and_expansion_caps(self):
        env = make_env("hybrid")
        result = env.step(Action(ActionKind.SEARCH, query="depression", k=999))
        self.assertEqual(result.charged_cost, 1)
        paper_id = next(item["id"] for item in env.observe().frontier if item["id"] == "W1")
        result = env.step(Action(ActionKind.EXPAND_REFS, paper_id=paper_id, k=999))
        self.assertLessEqual(len(result.observation.last_result["papers"]), 20)

    def test_insufficient_budget_does_not_go_negative(self):
        env = make_env("citation", budget=2)
        result = env.step(Action(ActionKind.EXPAND_REFS, paper_id="W1", k=20))
        self.assertEqual(result.error.code, "insufficient_budget")
        self.assertEqual(result.remaining_budget, 1)

    def test_internal_noop_is_charged(self):
        env = make_env("hybrid", budget=10)
        result = env.step(Action(ActionKind.NOOP))
        self.assertEqual(result.charged_cost, 1)
        self.assertEqual(result.remaining_budget, 8)


if __name__ == "__main__":
    unittest.main()

