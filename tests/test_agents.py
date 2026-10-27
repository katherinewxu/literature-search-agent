import json
import unittest
from pathlib import Path

from litsearch.agents import RankedRetrievalAgent, SnowballAgent
from litsearch.data import fixture_store
from litsearch.environment import LiteratureSearchEnv
from litsearch.llm import LLMSearchAgent, parse_turn
from litsearch.models import EpisodeSpec
from litsearch.retrieval import BM25Retriever
from litsearch.runner import run_episode


class ScriptedBackend:
    """Includes every unscreened paper it is shown, searches twice, then stops."""

    model = "scripted"

    def __init__(self):
        self.prompts = []

    def complete(self, messages, *, temperature, seed):
        user = messages[-1]["content"]
        self.prompts.append(user)
        unscreened = user.split("UNSCREENED PAPERS", 1)[1]
        ids = [line.split()[1] for line in unscreened.splitlines() if line.startswith("- W")]
        action = {"kind": "search", "query": "depression relapse"} if len(self.prompts) < 3 else {"kind": "stop"}
        return "```json\n" + json.dumps({"include": ids, "exclude": [], "action": action}) + "\n```"


class AgentTests(unittest.TestCase):
    def test_parse_plain_and_fenced_json(self):
        turn = parse_turn('{"include":["W1"],"action":{"kind":"read","paper_id":"W1"}}')
        self.assertEqual(turn["action"]["kind"], "read")
        turn = parse_turn('Sure.\n```json\n{"action":{"kind":"search","query":"x"}}\n```')
        self.assertEqual(turn["include"], [])
        self.assertEqual(parse_turn('{"action":"stop"}')["action"]["kind"], "stop")
        trailing = '{"include": [], "action": {"kind": "read", "paper_id": "W2"}}\n```\n\nWait.\n{"action": {"kind": "stop"}}'
        self.assertEqual(parse_turn(trailing)["action"]["kind"], "read")
        self.assertEqual(parse_turn('Let me think {about} it. {"action": {"kind": "stop"}}')["action"]["kind"], "stop")

    def test_parse_rejects_prose(self):
        with self.assertRaises(ValueError):
            parse_turn("I cannot choose")
        with self.assertRaises(ValueError):
            parse_turn('{"include": [], "action": {"kind": "teleport"}}')

    def test_llm_agent_screens_in_batches_and_records_turns(self):
        store = fixture_store()
        review = store.review("fixture_depression")
        spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, 10, "hybrid")
        backend = ScriptedBackend()
        agent = LLMSearchAgent(backend)
        run = run_episode(LiteratureSearchEnv(store, BM25Retriever(store.papers.values()), spec), agent)
        self.assertEqual(len(backend.prompts), 3)
        self.assertEqual(len(run["agent_turns"]), 3)
        self.assertGreater(run["metrics"]["included_count"], 0)
        self.assertEqual(run["agent_summary"]["parse_failures"], 0)
        for prompt in backend.prompts:
            self.assertNotIn("gold", prompt.lower())

    def test_llm_agent_charges_unknown_paper_as_noop(self):
        class BadBackend:
            model = "bad"

            def complete(self, messages, *, temperature, seed):
                return '{"action": {"kind": "read", "paper_id": "W_missing"}}'

        store = fixture_store()
        review = store.review("fixture_depression")
        spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, 4, "hybrid")
        run = run_episode(
            LiteratureSearchEnv(store, BM25Retriever(store.papers.values()), spec), LLMSearchAgent(BadBackend())
        )
        self.assertTrue(all(row["action"]["kind"] in {"seed", "noop", "stop"} for row in run["trace"]))
        self.assertGreater(run["agent_summary"]["invalid_ids"], 0)

    def test_spend_ledger_caps_across_backends(self):
        import tempfile

        from litsearch.llm import OpenAIChatBackend, SpendCapReached, SpendLedger

        ledger = SpendLedger(Path(tempfile.mkdtemp()) / "ledger.jsonl", cap_usd=1.0)
        ledger.add(0.6, "m", 10)
        ledger.check()
        SpendLedger(ledger.path, cap_usd=1.0).add(0.5, "m", 10)  # a second job, same file
        self.assertAlmostEqual(ledger.total(), 1.1)
        backend = OpenAIChatBackend("m", base_url="http://127.0.0.1:9/v1", ledger=ledger, retries=1)
        with self.assertRaises(SpendCapReached):
            backend.complete([{"role": "user", "content": "x"}], temperature=0, seed=None)

    def test_spend_ledger_is_thread_safe(self):
        import tempfile
        from concurrent.futures import ThreadPoolExecutor

        from litsearch.llm import SpendLedger

        path = Path(tempfile.mkdtemp()) / "ledger.jsonl"
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(lambda i: SpendLedger(path, 100.0).add(0.01, "m" * 200, i), range(400)))
        self.assertAlmostEqual(SpendLedger(path, 100.0).total(), 4.0)

    def test_disallowed_tool_is_charged_not_terminal(self):
        class SearchInCitationMode:
            model = "x"

            def complete(self, messages, *, temperature, seed):
                return '{"include": [], "action": {"kind": "search", "query": "q"}}'

        store = fixture_store()
        review = store.review("fixture_depression")
        spec = EpisodeSpec(review.id, review.question, review.eligibility_criteria, 4, "citation")
        run = run_episode(
            LiteratureSearchEnv(store, BM25Retriever(store.papers.values()), spec), LLMSearchAgent(SearchInCitationMode())
        )
        kinds = [row["action"]["kind"] for row in run["trace"]]
        self.assertNotIn("stop", kinds[:2])
        self.assertIn("noop", kinds)
        self.assertGreaterEqual(run["agent_summary"]["disallowed"], 1)

    def test_snowball_name(self):
        self.assertEqual(SnowballAgent().name, "snowball")
        self.assertEqual(RankedRetrievalAgent().name, "retrieval")


if __name__ == "__main__":
    unittest.main()
