import unittest

from litsearch.models import Action, ActionKind, EpisodeSpec, Paper, Review
from litsearch.store import CorpusStore
from litsearch.web import WebLiteratureSearchEnv, WebStore


def work(i, title, date="2015-01-01", doi=None, refs=()):
    return {
        "id": f"https://openalex.org/W{i}",
        "display_name": title,
        "publication_year": int(date[:4]),
        "publication_date": date,
        "doi": doi,
        "abstract_inverted_index": {"word": [0]},
        "referenced_works": [f"https://openalex.org/W{r}" for r in refs],
    }


class FakeClient:
    def __init__(self, works):
        self.works = works
        self.calls = []

    def _get(self, path, params):
        self.calls.append(params)
        page, per = int(params.get("page", 1)), int(params["per-page"])
        if "search" in params:
            pool = self.works
        else:  # citers query
            target = params["filter"].split(",")[0].split(":")[1]
            pool = [w for w in self.works if f"https://openalex.org/{target}" in w["referenced_works"]]
        return {"results": pool[(page - 1) * per : page * per]}

    def works_batch(self, ids, select=None):
        return [w for w in self.works if w["id"].rsplit("/", 1)[-1] in ids]


class WebEnvTest(unittest.TestCase):
    def setUp(self):
        gold = Paper(id="W1", title="A gold trial of ursodeoxycholic acid in biliary disease", doi="10.1/gold")
        review_paper = Paper(id="W900", title="Review title for leakage checks here")
        review = Review(
            id="r",
            question="Review title for leakage checks here",
            eligibility_criteria="RCTs",
            gold_ids=frozenset({"W1"}),
            search_cutoff_year=2018,
            review_paper_id="W900",
            search_cutoff_date="2018-06-01",
            candidate_ids=frozenset({"W1"}),
        )
        self.base = CorpusStore([gold, review_paper], [review])
        works = [
            work(500, "Preprint record of the gold trial", doi="10.1/GOLD"),  # same DOI as W1
            work(501, "Published after the cutoff", date="2019-01-01"),
            work(502, "Cites the review", refs=(900,)),
            work(503, "Review title for leakage checks here"),  # duplicate of the review
        ] + [work(600 + i, f"Unrelated paper number {i}") for i in range(60)]
        self.client = FakeClient(works)
        self.store = WebStore(self.base, self.client)

    def env(self, mode="hybrid", budget=10):
        spec = EpisodeSpec("r", "Review title for leakage checks here", "RCTs", budget, mode)
        return WebLiteratureSearchEnv(self.store, spec)

    def test_filters_and_canonical_gold_credit(self):
        env = self.env()
        seen = set(env.seed_ids)
        self.assertIn("W1", seen)  # W500 shown under the frozen store's id
        for hidden in ("W500", "W501", "W502", "W503", "W900"):
            self.assertNotIn(hidden, seen)
        env.step(Action(ActionKind.INCLUDE, paper_id="W1"))
        self.assertEqual(env.included, {"W1"})

    def test_repeated_query_pages_deeper(self):
        env = self.env()
        first = env.step(Action(ActionKind.SEARCH, query="unrelated", k=10)).observation.last_result["papers"]
        second = env.step(Action(ActionKind.SEARCH, query="unrelated", k=10)).observation.last_result["papers"]
        self.assertEqual(len(first), 10)
        self.assertEqual(len(second), 10)
        self.assertFalse({p["id"] for p in first} & {p["id"] for p in second})

    def test_citers_respect_cutoff_and_mode(self):
        env = self.env(mode="keyword")
        result = env.step(Action(ActionKind.EXPAND_CITERS, paper_id="W1"))
        self.assertEqual(result.error.code, "tool_unavailable")
        env = self.env(mode="citation")
        self.assertEqual(env.step(Action(ActionKind.SEARCH, query="x")).error.code, "tool_unavailable")


if __name__ == "__main__":
    unittest.main()
