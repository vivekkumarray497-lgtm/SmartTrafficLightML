def calculate_traffic_density(
    cars,
    motorcycles,
    buses,
    trucks,
    auto_rickshaws=0,
):
    total_vehicles = cars + motorcycles + buses + trucks + auto_rickshaws

    if total_vehicles == 0:
        density, level = "NO TRAFFIC", 0
    elif total_vehicles <= 5:
        density, level = "LOW", 1
    elif total_vehicles <= 12:
        density, level = "MEDIUM", 2
    elif total_vehicles <= 25:
        density, level = "HIGH", 3
    else:
        density, level = "VERY HIGH", 4

    return {
        "cars": cars,
        "motorcycles": motorcycles,
        "buses": buses,
        "trucks": trucks,
        "auto_rickshaws": auto_rickshaws,
        "total_vehicles": total_vehicles,
        "density": density,
        "level": level,
    }


def get_signal_from_density(density):
    if density in {"NO TRAFFIC", "LOW"}:
        return "GREEN"
    if density == "MEDIUM":
        return "YELLOW"
    if density in {"HIGH", "VERY HIGH"}:
        return "RED"
    return "RED"
