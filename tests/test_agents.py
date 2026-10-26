import unittest

from litsearch.agents import OpenAICompatibleAgent, RankedRetrievalAgent, SnowballAgent
from litsearch.models import ActionKind


class AgentTests(unittest.TestCase):
    def test_parse_plain_and_fenced_json(self):
        action = OpenAICompatibleAgent._parse('{"kind":"read","paper_id":"W1"}')
        self.assertEqual(action.kind, ActionKind.READ)
        action = OpenAICompatibleAgent._parse(
            '```json\n{"kind":"search","query":"depression","k":10}\n```'
        )
        self.assertEqual(action.kind, ActionKind.SEARCH)

    def test_parse_rejects_prose(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleAgent._parse("I cannot choose")

    def test_snowball_name(self):
        self.assertEqual(SnowballAgent().name, "snowball")
        self.assertEqual(RankedRetrievalAgent().name, "retrieval")


if __name__ == "__main__":
    unittest.main()

