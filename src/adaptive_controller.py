class AdaptiveController:
    DIRECTIONS = ("North", "South", "East", "West")

    def __init__(self):
        self.last_direction = None

    def decide(self, directions, pedestrians=0, emergency=False):
        directions = {d: max(0, int(directions.get(d, 0))) for d in self.DIRECTIONS}

        # Emergency mode is intentionally manual/demo controlled. The bundled
        # general YOLO model does not contain ambulance/fire-truck classes.
        if emergency:
            active = max(directions, key=directions.get) if any(directions.values()) else (self.last_direction or "North")
        else:
            scores = {d: directions[d] * 2 for d in self.DIRECTIONS}
            # Pedestrians increase priority slightly, but do not make an empty
            # road beat a heavily congested direction.
            for d in ("North", "South"):
                scores[d] += min(int(pedestrians), 5)
            active = max(scores, key=scores.get)

        self.last_direction = active
        result = {}
        for d, count in directions.items():
            green = max(15, min(70, 20 + count * 2))
            result[d] = {
                "signal": "Green" if d == active else "Red",
                "green_time": green if d == active else 0,
                "vehicles": count,
                "priority": count,
            }
        result["active_direction"] = active
        result["active_signal"] = "Green"
        result["cycle_seconds"] = max(30, result[active]["green_time"] + 10)
        return result
