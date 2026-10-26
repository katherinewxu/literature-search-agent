import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from litsearch.agents import RandomAgent
from litsearch.data import (
    audit_store,
    build_from_synergy_csv,
    build_from_synergy_source,
    fixture_store,
    load_review_metadata,
)
from litsearch.environment import LiteratureSearchEnv, allowed_ids_for
from litsearch.models import EpisodeSpec
from litsearch.retrieval import BM25Retriever
from litsearch.runner import run_episode


def _work(number, year, date, refs=(), abstract=True):
    return {
        "id": f"https://openalex.org/W{number}",
        "title": f"Study {number}",
        "abstract_inverted_index": {"fault": [0], "prediction": [1]} if abstract else None,
        "publication_year": year,
        "publication_date": date,
        "referenced_works": [f"https://openalex.org/W{ref}" for ref in refs],
        "primary_topic": {"field": {"display_name": "Computer Science"}},
    }


def _write_source(root: Path) -> None:
    review = root / "Hall_2011"
    review.mkdir(parents=True)
    with (review / "labels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["openalex_id", "doi", "label_included"])
        writer.writerow(["https://openalex.org/w1", "", "1"])
        writer.writerow(["https://openalex.org/w2", "", "0"])
        writer.writerow(["https://openalex.org/w3", "", "1"])  # dated after the review
        writer.writerow(["https://openalex.org/w4", "", "1"])  # absent from the works archive
    works = [
        _work(1, 2009, "2009-05-01", refs=(2,)),
        _work(2, 2008, "2008-01-01", abstract=False),
        _work(3, 2011, "2011-12-01"),
        _work(99, 2005, "2005-01-01"),  # unlabeled; must not enter the corpus
    ]
    with zipfile.ZipFile(review / "works_1.zip", "w") as bundle:
        bundle.writestr("works_0_3.json", json.dumps(works))
    (review / "metadata.json").write_text(
        json.dumps({"publication": {"eligibility_criteria": "Empirical fault prediction."}})
    )
    (review / "metadata_publication.json").write_text(
        json.dumps(
            {
                "id": "https://openalex.org/W500",
                "title": "Fault prediction review",
                "publication_year": 2011,
                "publication_date": "2011-10-11",
                "primary_topic": {"field": {"display_name": "Computer Science"}},
            }
        )
    )


class SynergySourceTests(unittest.TestCase):
    def test_metadata_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            exported = root / "review_metadata.csv"
            with exported.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, ["key", "title", "research_question"])
                writer.writeheader()
                writer.writerow(
                    {"key": "Hall_2011", "title": "CSV title", "research_question": "Which models?"}
                )
            overrides = root / "reviews.json"
            overrides.write_text(json.dumps({"Hall_2011": {"title": "Override title"}}))
            info = load_review_metadata(
                ["Hall_2011"], source_root=root, metadata_csvs=[exported], overrides=overrides
            )["Hall_2011"]
            self.assertEqual(info["title"], "Override title")
            self.assertNotIn("question", info)
            self.assertEqual(info["eligibility_criteria"], "Empirical fault prediction.")
            self.assertEqual(info["publication_date"], "2011-10-11")

    def test_source_build_keeps_unfiltered_pool_and_date_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_source(root)
            store, report = build_from_synergy_source(root, ["Hall_2011"])
            review = store.review("Hall_2011")
            self.assertEqual(set(store.papers), {"W1", "W2", "W3"})
            self.assertEqual(review.candidate_ids, frozenset({"W1", "W2", "W3"}))
            self.assertEqual(review.gold_ids, frozenset({"W1", "W3"}))
            self.assertEqual(review.question, "Fault prediction review")
            self.assertEqual(review.review_paper_id, "W500")
            self.assertEqual(review.search_cutoff_year, 2011)
            self.assertEqual(review.search_cutoff_date, "2011-10-11")
            self.assertEqual(review.domain, "Computer Science")
            self.assertEqual(report["Hall_2011"]["missing_gold"], ["W4"])
            self.assertEqual(store.paper("W1").abstract, "fault prediction")
            self.assertEqual(store.paper("W2").abstract, "")
            self.assertEqual(store.paper("W2").citers, ("W1",))
            self.assertEqual(allowed_ids_for(store, review), {"W1", "W2"})
            audit = audit_store(store, min_gold=2)
            self.assertTrue(audit["ok"])
            self.assertEqual(
                [w["paper_id"] for w in audit["warnings"] if w["code"] == "post_cutoff_gold"],
                ["W3"],
            )
            self.assertEqual(audit["reviews"]["Hall_2011"]["visible_gold"], 1)

    def test_csv_build_ignores_metadata_folder_and_uses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            with (root / "metadata" / "review_metadata.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, ["key", "title", "publication_year", "publication_date", "openalex_id"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "key": "Example_2020",
                        "title": "Example review",
                        "publication_year": "2020",
                        "publication_date": "2020-03-01",
                        "openalex_id": "https://openalex.org/W77",
                    }
                )
            with (root / "Example_2020.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, ["openalex_id", "title", "label_included"])
                writer.writeheader()
                writer.writerow({"openalex_id": "W1", "title": "A", "label_included": "1"})
            store = build_from_synergy_csv(root)
            self.assertEqual(set(store.papers), {"W1"})
            review = store.review("Example_2020")
            self.assertEqual(review.question, "Example review")
            self.assertEqual(review.review_paper_id, "W77")
            self.assertEqual(review.search_cutoff_date, "2020-03-01")


class SplitTests(unittest.TestCase):
    def test_single_review_domains_still_produce_a_test_split(self):
        from dataclasses import replace
        from litsearch.data import deterministic_split

        base = fixture_store().review("fixture_depression")
        reviews = [replace(base, id=f"R{i}", domain=f"D{i}") for i in range(8)]
        splits = deterministic_split(reviews)
        self.assertEqual([len(splits[k]) for k in ("train", "development", "test")], [4, 2, 2])
        self.assertEqual(sorted(sum(splits.values(), [])), sorted(r.id for r in reviews))


class BaselineBehaviourTests(unittest.TestCase):
    def test_bm25_matches_reference_formula(self):
        store = fixture_store()
        retriever = BM25Retriever(store.papers.values())
        query = "prospective relapse depression relapse"
        from collections import Counter
        from litsearch.retrieval import tokenize

        for paper in store.papers.values():
            counts = Counter(tokenize(f"{paper.title} {paper.title} {paper.abstract}"))
            length = sum(counts.values())
            expected = 0.0
            for term in tokenize(query):
                tf = counts.get(term, 0)
                if tf:
                    norm = tf + 1.5 * (0.25 + 0.75 * length / retriever.avg_length)
                    expected += retriever.idf[term] * tf * 2.5 / norm
            self.assertAlmostEqual(retriever.score(query, paper.id), expected)

    def test_random_agent_includes_what_it_reads(self):
        store = fixture_store()
        review = store.review("fixture_depression")
        spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, 20, "hybrid")
        run = run_episode(
            LiteratureSearchEnv(store, BM25Retriever(store.papers.values()), spec), RandomAgent(seed=3)
        )
        self.assertGreater(run["metrics"]["included_count"], 0)
        # The final read may exhaust the budget, ending the episode before its free include.
        self.assertGreaterEqual(run["metrics"]["included_count"], run["metrics"]["unique_read"] - 1)


if __name__ == "__main__":
    unittest.main()
