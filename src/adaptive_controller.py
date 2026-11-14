class AdaptiveController:
    DIRECTIONS = ("North", "South", "East", "West")

    def __init__(self):
        self.last_direction = None

    def decide(
        self,
        directions,
        pedestrians=0,
        emergency=False,
        emergency_direction=None,
        pedestrian_directions=None,
    ):
        directions = {
            direction: max(0, int(directions.get(direction, 0)))
            for direction in self.DIRECTIONS
        }

        pedestrian_directions = pedestrian_directions or {}
        directional_pedestrians = {
            direction: max(0, int(pedestrian_directions.get(direction, 0)))
            for direction in self.DIRECTIONS
        }

        if emergency and emergency_direction in self.DIRECTIONS:
            active = emergency_direction
        elif emergency:
            active = max(
                self.DIRECTIONS,
                key=lambda direction: directions[direction],
            ) if any(directions.values()) else (self.last_direction or "North")
        else:
            scores = {
                direction: directions[direction] * 2
                + min(directional_pedestrians[direction], 5)
                for direction in self.DIRECTIONS
            }
            active = max(
                self.DIRECTIONS,
                key=lambda direction: (
                    scores[direction],
                    direction == self.last_direction,
                ),
            )

        self.last_direction = active
        result = {}
        for direction, count in directions.items():
            green = max(15, min(70, 20 + count * 2))
            result[direction] = {
                "signal": "Green" if direction == active else "Red",
                "green_time": (
                    60 if emergency and direction == active
                    else (green if direction == active else 0)
                ),
                "vehicles": count,
                "pedestrians": directional_pedestrians[direction],
                "priority": (
                    count * 2 + min(directional_pedestrians[direction], 5)
                ),
            }
        result["active_direction"] = active
        result["active_signal"] = "Green"
        result["cycle_seconds"] = (
            60 if emergency else max(30, result[active]["green_time"] + 10)
        )
        return result
