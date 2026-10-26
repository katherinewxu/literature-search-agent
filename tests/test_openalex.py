import unittest

from litsearch.data import fixture_store
from litsearch.openalex import abstract_from_inverted, backfill_abstracts, paper_from_work, short_id


class OpenAlexTests(unittest.TestCase):
    def test_abstract_reconstruction(self):
        index = {"world": [1], "Hello": [0, 2]}
        self.assertEqual(abstract_from_inverted(index), "Hello world Hello")

    def test_work_conversion(self):
        paper = paper_from_work(
            {
                "id": "https://openalex.org/W123",
                "display_name": "Study",
                "publication_year": 2020,
                "abstract_inverted_index": {"Text": [0]},
                "referenced_works": ["https://openalex.org/W2"],
                "doi": "https://doi.org/10.1/test",
            },
            citers=["W4"],
        )
        self.assertEqual(paper.id, "W123")
        self.assertEqual(paper.references, ("W2",))
        self.assertEqual(paper.citers, ("W4",))
        self.assertEqual(short_id(None), None)

    def test_backfill_fills_only_empty_abstracts(self):
        class FakeClient:
            def __init__(self):
                self.requests = []

            def works_batch(self, ids):
                self.requests.append(list(ids))
                return [
                    {"id": f"https://openalex.org/{i}", "abstract_inverted_index": {"New": [0]}}
                    for i in ids
                ]

        store = fixture_store()
        client = FakeClient()
        filled, stats = backfill_abstracts(store, client, batch_size=2)
        self.assertEqual(stats["missing_before"], 0)
        self.assertEqual(client.requests, [])
        from dataclasses import replace
        from litsearch.store import CorpusStore

        papers = [replace(p, abstract="") if p.id in {"W7", "W8", "W9"} else p for p in store.papers.values()]
        filled, stats = backfill_abstracts(CorpusStore(papers, store.reviews.values()), client, batch_size=2)
        self.assertEqual(client.requests, [["W7", "W8"], ["W9"]])
        self.assertEqual(stats["filled"], 3)
        self.assertEqual(stats["missing_after"], 0)
        self.assertEqual(filled.paper("W7").abstract, "New")
        self.assertEqual(filled.paper("W1").abstract, store.paper("W1").abstract)


if __name__ == "__main__":
    unittest.main()

