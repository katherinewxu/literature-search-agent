import tempfile
import unittest
from pathlib import Path

from litsearch import analysis
from litsearch.agents import RankedRetrievalAgent, SnowballAgent
from litsearch.analysis_report import write_analysis
from litsearch.data import fixture_store, write_artifacts
from litsearch.environment import LiteratureSearchEnv
from litsearch.models import EpisodeSpec
from litsearch.retrieval import BM25Retriever
from litsearch.rl import episode_reward
from litsearch.runner import run_episode, write_run


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = fixture_store()
        write_artifacts(self.store, self.tmp / "data", "fixture")
        review = self.store.review("fixture_depression")
        for agent_factory, mode in ((RankedRetrievalAgent, "keyword"), (SnowballAgent, "hybrid")):
            for budget in (4, 8):
                spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, budget, mode)
                agent = agent_factory()
                run = run_episode(LiteratureSearchEnv(self.store, BM25Retriever(self.store.papers.values()), spec), agent)
                write_run(run, self.tmp / "runs" / f"{agent.name}_{mode}_{budget}.json")

    def test_series_are_monotone_and_bounded(self):
        runs = analysis.load_runs([self.tmp / "runs"])
        self.assertEqual(len(runs), 4)
        gold = self.store.review("fixture_depression").gold_ids
        for run in runs:
            series = analysis.run_series(run, gold)
            self.assertEqual(series["cost"], sorted(series["cost"]))
            for value, exposed in zip(series["recall"], series["exposure_recall"]):
                self.assertLessEqual(value, exposed + 1e-9)  # can only include what was exposed
            self.assertAlmostEqual(series["recall"][-1], run["metrics"]["recall"])

    def test_reachability_counts_seeds_at_zero_hops(self):
        seeds = analysis.seeds_for(self.store, "fixture_depression")
        result = analysis.reachability(self.store, "fixture_depression", seeds)
        gold_seeds = len(set(seeds) & self.store.review("fixture_depression").gold_ids)
        self.assertAlmostEqual(result["reachable_within"][0], gold_seeds / result["gold"])
        self.assertEqual(sorted(result["reachable_within"].values()), list(result["reachable_within"].values()))

    def test_report_writes_markdown(self):
        path = write_analysis(self.tmp / "data", [self.tmp / "runs"], self.tmp / "analysis")
        text = Path(path).read_text()
        self.assertIn("Search vs. screening", text)
        self.assertIn("retrieval/keyword", text)

    def test_reward_penalizes_false_inclusions(self):
        metrics = {"recall": 0.5, "included_count": 10, "true_positives": 5, "charged_cost": 20}
        self.assertAlmostEqual(episode_reward(metrics, 20, lam=0.2), 0.5 - 0.2 * 0.5)
        self.assertAlmostEqual(episode_reward({**metrics, "included_count": 0, "true_positives": 0, "recall": 0}, 20), 0.0)


if __name__ == "__main__":
    unittest.main()
