import unittest

from src.traffic_density import calculate_traffic_density


class TrafficDensityTests(unittest.TestCase):
    def test_zero_vehicles_is_no_traffic(self):
        result = calculate_traffic_density(0, 0, 0, 0)
        self.assertEqual(result["density"], "NO TRAFFIC")
        self.assertEqual(result["total_vehicles"], 0)

    def test_density_bands_are_consistent(self):
        cases = [
            ((2, 1, 1, 1), "LOW"),
            ((5, 4, 2, 1), "MEDIUM"),
            ((10, 8, 4, 3), "HIGH"),
            ((15, 10, 5, 2), "VERY HIGH"),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(
                    calculate_traffic_density(*values)["density"],
                    expected,
                )

    def test_auto_rickshaws_are_included_in_total_and_density(self):
        result = calculate_traffic_density(1, 1, 0, 0, auto_rickshaws=4)
        self.assertEqual(result["auto_rickshaws"], 4)
        self.assertEqual(result["total_vehicles"], 6)
        self.assertEqual(result["density"], "MEDIUM")


if __name__ == "__main__":
    unittest.main()
