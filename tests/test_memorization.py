import unittest

from litsearch.memorization import parse_titles, score_review, title_match
from litsearch.models import Paper, Review
from litsearch.store import CorpusStore


class MemorizationTest(unittest.TestCase):
    def test_parse_and_match(self):
        text = 'Sure:\n[{"title": "Ursodeoxycholic acid in primary biliary cirrhosis: a trial", "year": 1994}]'
        self.assertEqual(parse_titles(text), ["Ursodeoxycholic acid in primary biliary cirrhosis: a trial"])
        self.assertGreater(title_match("Ursodeoxycholic acid in PBC", "ursodeoxycholic acid in pbc."), 0.95)
        self.assertLess(title_match("Exercise and depression", "Ursodeoxycholic acid in PBC"), 0.5)

    def test_score_counts_gold_only_through_pool(self):
        papers = [
            Paper(id="W1", title="A randomised trial of ursodeoxycholic acid in primary biliary cirrhosis"),
            Paper(id="W2", title="Bezafibrate for pruritus in cholestatic liver disease patients"),
        ]
        review = Review("r", "q", "c", frozenset({"W1"}), 2020, candidate_ids=frozenset({"W1", "W2"}))
        store = CorpusStore(papers, [review])
        row = score_review(
            store,
            "r",
            [
                "A randomized trial of ursodeoxycholic acid in primary biliary cirrhosis",
                "Bezafibrate for pruritus in cholestatic liver disease patients",
                "Some hallucinated title about something else entirely",
            ],
        )
        self.assertEqual(row["matched_gold"], 1)
        self.assertEqual(row["matched_pool"], 2)
        self.assertAlmostEqual(row["closed_book_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
