import unittest

from litsearch.recall_estimation import cp_upper, saturation_estimate


class RecallEstimationTest(unittest.TestCase):
    def test_cp_upper(self):
        self.assertAlmostEqual(cp_upper(0, 80, 0.05), 1 - 0.05 ** (1 / 80), places=4)
        self.assertGreater(cp_upper(5, 80), 5 / 80)
        self.assertEqual(cp_upper(3, 3), 1.0)

    def test_saturation(self):
        import math
        trace = [{"charged_cost": 1, "included_ids": [str(i) for i in range(round(50 * (1 - math.exp(-0.1 * t))))]}
                 for t in range(1, 41)]
        est = saturation_estimate({"trace": trace})
        self.assertAlmostEqual(est, 49 / 50, delta=0.05)


if __name__ == "__main__":
    unittest.main()
