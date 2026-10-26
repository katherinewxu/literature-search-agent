import unittest

from litsearch.models import Action, ActionKind, EpisodeSpec
from litsearch.store import normalize_openalex_id


class ModelTests(unittest.TestCase):
    def test_action_round_trip(self):
        action = Action.from_dict({"kind": "search", "query": "depression", "k": 10})
        self.assertEqual(action.kind, ActionKind.SEARCH)
        self.assertEqual(action.to_dict()["query"], "depression")

    def test_action_validation(self):
        with self.assertRaises(ValueError):
            Action.from_dict({"kind": "read"})
        with self.assertRaises(ValueError):
            Action.from_dict({"kind": "search", "query": ""})
        with self.assertRaises(ValueError):
            Action.from_dict({"kind": "unknown"})

    def test_episode_validation(self):
        with self.assertRaises(ValueError):
            EpisodeSpec("r", "q", "e", 0, "hybrid")

    def test_openalex_normalization(self):
        self.assertEqual(normalize_openalex_id("https://openalex.org/W123/"), "W123")
        self.assertEqual(normalize_openalex_id("w9"), "W9")
        with self.assertRaises(ValueError):
            normalize_openalex_id("doi:123")


if __name__ == "__main__":
    unittest.main()

