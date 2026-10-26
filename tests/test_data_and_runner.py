import json
import csv
import tempfile
import unittest
from pathlib import Path

from litsearch.agents import SnowballAgent
from litsearch.data import audit_store, fixture_store, write_artifacts
from litsearch.data import build_from_synergy_csv
from litsearch.demo import build_html_replay
from litsearch.environment import LiteratureSearchEnv
from litsearch.models import EpisodeSpec
from litsearch.retrieval import BM25Retriever
from litsearch.runner import run_episode, write_run
from litsearch.store import CorpusStore


class EndToEndTests(unittest.TestCase):
    def test_synergy_extended_csv_reconstructs_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "openalex_id": "https://openalex.org/w1",
                    "title": "Included study",
                    "abstract": "Text",
                    "publication_year": "2018",
                    "label_included": "1",
                    "referenced_works": "['https://openalex.org/W2']",
                    "primary_topic_domain": "Medicine",
                },
                {
                    "openalex_id": "W2",
                    "title": "Earlier study",
                    "abstract": "Text",
                    "publication_year": "2015",
                    "label_included": "0",
                    "referenced_works": "[]",
                    "primary_topic_domain": "Medicine",
                },
            ]
            path = root / "Example_2020.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            store = build_from_synergy_csv(root)
            self.assertEqual(store.paper("W1").references, ("W2",))
            self.assertEqual(store.paper("W2").citers, ("W1",))
            self.assertEqual(store.review("Example_2020").search_cutoff_year, 2019)
            self.assertEqual(store.review("Example_2020").domain, "Medicine")

    def test_artifacts_episode_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = fixture_store()
            manifest = write_artifacts(store, root / "data", "fixture")
            self.assertEqual(manifest["review_count"], 1)
            loaded = CorpusStore.load(root / "data")
            self.assertTrue(audit_store(loaded)["ok"])
            review = loaded.review("fixture_depression")
            env = LiteratureSearchEnv(
                loaded,
                BM25Retriever(loaded.papers.values()),
                EpisodeSpec(review.id, review.question, review.eligibility_criteria, 20, "hybrid"),
            )
            run = run_episode(env, SnowballAgent())
            self.assertGreaterEqual(run["metrics"]["recall"], 0.75)
            path = write_run(run, root / "run.json")
            self.assertTrue(path.with_suffix(".json.sha256").exists())
            html = build_html_replay(json.loads(path.read_text()))
            self.assertIn("Literature-search replay", html)
            self.assertNotIn("https://", html)


if __name__ == "__main__":
    unittest.main()

