class AdaptiveController:
    DIRECTIONS = ("North", "South", "East", "West")

    def __init__(self):
        self.last_direction = None

    def decide(self, directions, pedestrians=0, emergency=False, emergency_direction=None):
        directions = {d: max(0, int(directions.get(d, 0))) for d in self.DIRECTIONS}

        if emergency and emergency_direction in self.DIRECTIONS:
            active = emergency_direction
        elif emergency:
            active = max(directions, key=directions.get) if any(directions.values()) else (self.last_direction or "North")
        else:
            scores = {d: directions[d] * 2 for d in self.DIRECTIONS}
            for d in ("North", "South"):
                scores[d] += min(int(pedestrians), 5)
            active = max(scores, key=scores.get)

        self.last_direction = active
        result = {}
        for d, count in directions.items():
            green = max(15, min(70, 20 + count * 2))
            result[d] = {
                "signal": "Green" if d == active else "Red",
                "green_time": 60 if emergency and d == active else (green if d == active else 0),
                "vehicles": count,
                "priority": count,
            }
        result["active_direction"] = active
        result["active_signal"] = "Green"
        result["cycle_seconds"] = 60 if emergency else max(30, result[active]["green_time"] + 10)
        return result
