import unittest

from litsearch.rl import episode_reward, returns_to_go, turn_gains


def ev(kind, result=(), included=()):
    return {"action": {"kind": kind}, "result_ids": list(result), "included_ids": sorted(included)}


class TurnCreditTest(unittest.TestCase):
    def test_segments_and_gains(self):
        gold = {"g1", "g2"}
        run = {
            "seed_ids": ["s1", "g1"],
            "trace": [
                {"action": {"kind": "seed"}, "result_ids": ["s1", "g1"]},
                ev("include", included={"g1"}),          # turn 1: include gold seen in seeds
                ev("search", result=["g2", "x1"], included={"g1"}),  # turn 1 ends: discovers g2
                ev("include", included={"g1", "x1"}),     # turn 2: false positive
                ev("expand_citers", result=["g2", "x2"], included={"g1", "x1"}),  # redundant g2
                ev("stop", included={"g1", "x1"}),        # turn 3
            ],
            "agent_turns": [{}, {}, {}],
        }
        gains = turn_gains(run, gold, fp_weight=0.5, discovery_weight=0.5, redundancy_weight=0.1)
        self.assertEqual(len(gains), 3)
        self.assertAlmostEqual(gains[0], 0.5 + 0.25)          # +1/2 included, +0.5*1/2 discovered
        self.assertAlmostEqual(gains[1], -0.25 - 0.1 * 0.5)   # FP, half of results already seen
        self.assertAlmostEqual(gains[2], 0.0)
        self.assertEqual(returns_to_go([1.0, 2.0, 3.0]), [6.0, 5.0, 3.0])

    def test_misaligned_returns_none(self):
        run = {"seed_ids": [], "trace": [ev("search", result=["a"])], "agent_turns": [{}, {}]}
        self.assertIsNone(turn_gains(run, {"a"}))

    def test_fbeta(self):
        m = {"precision": 0.5, "recall": 0.5, "included_count": 4, "true_positives": 2, "charged_cost": 10}
        self.assertAlmostEqual(episode_reward(m, 20, kind="fbeta", beta=2.0), 0.5)
        self.assertAlmostEqual(episode_reward(m, 20, lam=0.1), 0.45)


if __name__ == "__main__":
    unittest.main()
