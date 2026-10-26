import unittest

from litsearch.metrics import area_under_recall_cost, compute_metrics
from litsearch.models import TraceEvent


class MetricTests(unittest.TestCase):
    def test_metrics_and_censored_wss(self):
        events = [
            TraceEvent(1, {"kind": "read"}, 1, 1, included_ids=[]),
            TraceEvent(2, {"kind": "include"}, 0, 1, included_ids=["W1"]),
        ]
        metrics = compute_metrics(
            included={"W1"},
            gold={"W1", "W2"},
            trace=events,
            budget=2,
            exposed={"W1"},
            read_ids={"W1"},
            candidate_count=10,
        )
        self.assertEqual(metrics.recall, 0.5)
        self.assertIsNone(metrics.wss_at_95)

    def test_aurc_is_normalized(self):
        events = [TraceEvent(1, {"kind": "read"}, 1, 1, included_ids=["W1"])]
        value = area_under_recall_cost(events, {"W1"}, budget=2)
        self.assertEqual(value, 0.5)


if __name__ == "__main__":
    unittest.main()

