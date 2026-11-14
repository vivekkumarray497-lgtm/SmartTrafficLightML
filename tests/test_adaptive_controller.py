import unittest

from src.adaptive_controller import AdaptiveController


class AdaptiveControllerTests(unittest.TestCase):
    def test_pedestrian_count_increases_only_its_own_approach_priority(self):
        controller = AdaptiveController()
        result = controller.decide(
            directions={"North": 0, "South": 0, "East": 2, "West": 0},
            pedestrian_directions={"North": 0, "South": 0, "East": 3, "West": 0},
        )
        self.assertEqual(result["active_direction"], "East")
        self.assertEqual(result["East"]["pedestrians"], 3)
        self.assertEqual(result["North"]["pedestrians"], 0)
        self.assertEqual(result["South"]["pedestrians"], 0)

    def test_pedestrians_without_direction_are_not_assigned_to_north_and_south(self):
        controller = AdaptiveController()
        result = controller.decide(
            directions={"North": 0, "South": 0, "East": 2, "West": 0},
            pedestrians=20,
        )
        self.assertEqual(result["active_direction"], "East")
        self.assertEqual(result["North"]["pedestrians"], 0)
        self.assertEqual(result["South"]["pedestrians"], 0)

    def test_emergency_direction_overrides_normal_traffic_priority(self):
        controller = AdaptiveController()
        result = controller.decide(
            directions={"North": 20, "South": 0, "East": 0, "West": 0},
            emergency=True,
            emergency_direction="West",
        )
        self.assertEqual(result["active_direction"], "West")
        self.assertEqual(result["West"]["signal"], "Green")
        self.assertEqual(result["North"]["signal"], "Red")


if __name__ == "__main__":
    unittest.main()
