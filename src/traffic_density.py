def calculate_traffic_density(
    cars,
    motorcycles,
    buses,
    trucks
):
    total_vehicles = (
        cars +
        motorcycles +
        buses +
        trucks
    )

    if total_vehicles == 0:
        density = "NO TRAFFIC"
        level = 0

    elif total_vehicles <= 5:
        density = "LOW"
        level = 1

    elif total_vehicles <= 12:
        density = "MEDIUM"
        level = 2

    elif total_vehicles <= 25:
        density = "HIGH"
        level = 3

    else:
        density = "VERY HIGH"
        level = 4

    return {
        "cars": cars,
        "motorcycles": motorcycles,
        "buses": buses,
        "trucks": trucks,
        "total_vehicles": total_vehicles,
        "density": density,
        "level": level
    }


def get_signal_from_density(density):
    if density == "NO TRAFFIC":
        return "GREEN"

    elif density == "LOW":
        return "GREEN"

    elif density == "MEDIUM":
        return "YELLOW"

    elif density == "HIGH":
        return "RED"

    elif density == "VERY HIGH":
        return "RED"

    return "RED"


if __name__ == "__main__":

    print("Smart Traffic Density System")
    print()

    test_cases = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (5, 4, 1, 1),
        (10, 7, 2, 2),
        (20, 10, 3, 2)
    ]

    for cars, motorcycles, buses, trucks in test_cases:

        result = calculate_traffic_density(
            cars,
            motorcycles,
            buses,
            trucks
        )

        signal = get_signal_from_density(
            result["density"]
        )

        print(result)
        print("Signal:", signal)
        print()