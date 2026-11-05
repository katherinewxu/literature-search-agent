import unittest

from litsearch.forced import ForcedExpansionAgent
from litsearch.models import Action, ActionKind, Observation


class AlwaysSearch:
    name = "inner"

    def __init__(self):
        self._history = []
        self.turns = []

    def act(self, observation):
        return Action(ActionKind.SEARCH, query="q", k=10)


def obs(included=(), mode="hybrid"):
    return Observation(question="q", eligibility_criteria="c", remaining_budget=10, mode=mode,
                       included_ids=tuple(included), excluded_ids=(), frontier=(), last_result={}, action_count=0)


class ForcedTest(unittest.TestCase):
    def test_every_third_charged_action_expands_latest_inclusion(self):
        agent = ForcedExpansionAgent(AlwaysSearch(), every=3)
        kinds = [agent.act(obs(included=("A",))).kind for _ in range(6)]
        self.assertEqual(kinds[:2], [ActionKind.SEARCH, ActionKind.SEARCH])
        self.assertEqual(kinds[2], ActionKind.EXPAND_CITERS)
        self.assertEqual(kinds[5], ActionKind.SEARCH)  # A already expanded, nothing new included
        self.assertEqual(agent.summary()["forced_expansions"], 1)
        self.assertEqual(agent.turns, [])

    def test_no_forcing_outside_hybrid(self):
        agent = ForcedExpansionAgent(AlwaysSearch(), every=1)
        self.assertEqual(agent.act(obs(included=("A",), mode="keyword")).kind, ActionKind.SEARCH)


if __name__ == "__main__":
    unittest.main()
