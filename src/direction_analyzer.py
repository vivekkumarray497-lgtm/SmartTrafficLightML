def analyze_direction(
    cars,
    motorcycles,
    buses,
    trucks
):
    total = cars + motorcycles + buses + trucks

    if total == 0:
        density = "NO TRAFFIC"
    elif total <= 5:
        density = "LOW"
    elif total <= 12:
        density = "MEDIUM"
    elif total <= 25:
        density = "HIGH"
    else:
        density = "VERY HIGH"

    return {
        "cars": cars,
        "motorcycles": motorcycles,
        "buses": buses,
        "trucks": trucks,
        "total": total,
        "density": density
    }


def analyze_all_directions(direction_data):

    result = {}

    for direction, data in direction_data.items():

        result[direction] = analyze_direction(
            data.get("cars", 0),
            data.get("motorcycles", 0),
            data.get("buses", 0),
            data.get("trucks", 0)
        )

    return result


def get_priority_direction(direction_data):

    highest_direction = None
    highest_traffic = -1

    for direction, data in direction_data.items():

        total = (
            data.get("cars", 0)
            + data.get("motorcycles", 0)
            + data.get("buses", 0)
            + data.get("trucks", 0)
        )

        if total > highest_traffic:
            highest_traffic = total
            highest_direction = direction

    return highest_direction


if __name__ == "__main__":

    direction_data = {

        "North": {
            "cars": 30,
            "motorcycles": 15,
            "buses": 2,
            "trucks": 4
        },

        "South": {
            "cars": 8,
            "motorcycles": 5,
            "buses": 0,
            "trucks": 1
        },

        "East": {
            "cars": 12,
            "motorcycles": 7,
            "buses": 1,
            "trucks": 1
        },

        "West": {
            "cars": 20,
            "motorcycles": 10,
            "buses": 2,
            "trucks": 3
        }
    }

    results = analyze_all_directions(direction_data)

    print("Smart Traffic Direction Analyzer")
    print()

    for direction, data in results.items():

        print(
            f"{direction}: "
            f"Cars={data['cars']}, "
            f"Bikes={data['motorcycles']}, "
            f"Buses={data['buses']}, "
            f"Trucks={data['trucks']}, "
            f"Total={data['total']}, "
            f"Density={data['density']}"
        )

    priority = get_priority_direction(direction_data)

    print()
    print("Traffic Priority Direction:", priority)