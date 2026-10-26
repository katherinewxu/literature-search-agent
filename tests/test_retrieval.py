import unittest

from litsearch.data import fixture_store
from litsearch.retrieval import BM25Retriever, tokenize


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.store = fixture_store()
        self.retriever = BM25Retriever(self.store.papers.values())

    def test_tokenize(self):
        self.assertEqual(tokenize("Major-depression, 2024!"), ["major", "depression", "2024"])

    def test_relevant_papers_rank_first(self):
        ids = self.retriever.search("prospective predictors depressive relapse", 4)
        self.assertIn("W3", ids[:2])

    def test_allowed_ids(self):
        ids = self.retriever.search("depression", 10, {"W3", "W8"})
        self.assertEqual(ids, ["W3"])


if __name__ == "__main__":
    unittest.main()

