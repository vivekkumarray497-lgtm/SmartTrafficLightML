def calculate_green_time(total_vehicles):
    if total_vehicles <= 0:
        return 15
    elif total_vehicles <= 5:
        return 20
    elif total_vehicles <= 12:
        return 30
    elif total_vehicles <= 25:
        return 45
    else:
        return 60


def get_density(total_vehicles):
    if total_vehicles == 0:
        return "NO TRAFFIC"
    elif total_vehicles <= 5:
        return "LOW"
    elif total_vehicles <= 12:
        return "MEDIUM"
    elif total_vehicles <= 25:
        return "HIGH"
    else:
        return "VERY HIGH"


def get_signal_message(total_vehicles, emergency=False):

    if emergency:
        return {
            "signal": "GREEN",
            "message": "EMERGENCY PRIORITY - GREEN LIGHT",
            "priority": True,
            "density": get_density(total_vehicles),
            "green_time": 60
        }

    density = get_density(total_vehicles)

    if density == "NO TRAFFIC":
        signal = "GREEN"
        message = "No traffic - GREEN LIGHT"

    elif density == "LOW":
        signal = "GREEN"
        message = "Traffic is low - GREEN LIGHT"

    elif density == "MEDIUM":
        signal = "GREEN"
        message = "Moderate traffic - ADAPTIVE GREEN"

    elif density == "HIGH":
        signal = "GREEN"
        message = "Heavy traffic - LONGER GREEN TIME"

    else:
        signal = "GREEN"
        message = "Very heavy traffic - MAXIMUM GREEN TIME"

    return {
        "signal": signal,
        "message": message,
        "priority": False,
        "density": density,
        "green_time": calculate_green_time(total_vehicles)
    }


def get_direction_signal(direction_data, emergency_direction=None):

    result = {}

    for direction, data in direction_data.items():

        total = (
            data.get("cars", 0)
            + data.get("motorcycles", 0)
            + data.get("buses", 0)
            + data.get("trucks", 0)
        )

        result[direction] = {
            "vehicles": total,
            "density": get_density(total),
            "signal": "RED",
            "green_time": 0,
            "priority": False
        }

    if emergency_direction in result:

        result[emergency_direction]["signal"] = "GREEN"
        result[emergency_direction]["green_time"] = 60
        result[emergency_direction]["priority"] = True

        return result

    priority_direction = None
    highest_traffic = -1

    for direction, data in result.items():

        if data["vehicles"] > highest_traffic:
            highest_traffic = data["vehicles"]
            priority_direction = direction

    if priority_direction is not None:

        result[priority_direction]["signal"] = "GREEN"

        result[priority_direction]["green_time"] = calculate_green_time(
            highest_traffic
        )

    return result


if __name__ == "__main__":

    print("Smart Traffic Signal Controller")
    print()

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

    print("Normal Traffic")
    print()

    signals = get_direction_signal(
        direction_data
    )

    for direction, data in signals.items():

        print(
            direction,
            "=>",
            data
        )

    print()
    print("Emergency Traffic")
    print()

    emergency_signals = get_direction_signal(
        direction_data,
        emergency_direction="East"
    )

    for direction, data in emergency_signals.items():

        print(
            direction,
            "=>",
            data
        )