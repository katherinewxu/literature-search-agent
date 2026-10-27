import unittest

from litsearch.environment import allowed_ids_for
from litsearch.models import Paper, Review
from litsearch.open_graph import build_open_graph, open_graph_candidates
from litsearch.store import CorpusStore


class FakeClient:
    def __init__(self):
        self.requests = []

    def works_batch(self, ids, select=None):
        self.requests.append(list(ids))
        works = {
            "W10": {"id": "https://openalex.org/W10", "title": "Bridge", "publication_year": 2001,
                    "referenced_works": ["https://openalex.org/W2"]},
            "W11": {"id": "https://openalex.org/W11", "title": "Late", "publication_year": 2030},
        }
        return [works[i] for i in ids if i in works]


def make_store():
    papers = [
        Paper("W1", "Gold study", publication_year=2005, references=("W10", "W11")),
        Paper("W2", "Other", publication_year=2004, references=("W10",)),
        Paper("W3", "Not in pool", publication_year=2004),
    ]
    review = Review("r", "q", "c", frozenset({"W1"}), 2010, candidate_ids=frozenset({"W1", "W2"}))
    return CorpusStore(papers, [review])


class OpenGraphTests(unittest.TestCase):
    def test_candidates_ranked_by_pool_citations(self):
        store = make_store()
        self.assertEqual(open_graph_candidates(store, store.review("r"), 10), ["W10", "W11"])
        self.assertEqual(open_graph_candidates(store, store.review("r"), 1), ["W10"])

    def test_build_adds_neighbors_and_keeps_leakage_filters(self):
        client = FakeClient()
        opened, stats = build_open_graph(make_store(), client, cap_per_review=10, pool_multiple=5)
        review = opened.review("r")
        self.assertEqual(review.gold_ids, frozenset({"W1"}))
        self.assertEqual(review.candidate_ids, frozenset({"W1", "W2", "W10", "W11"}))
        self.assertEqual(stats["fetched"], 2)
        self.assertIn("W10", opened.paper("W2").citers)  # new reverse edge
        visible = allowed_ids_for(opened, review)
        self.assertIn("W10", visible)
        self.assertNotIn("W11", visible)  # post-cutoff work stays hidden
        self.assertNotIn("W3", visible)


if __name__ == "__main__":
    unittest.main()
