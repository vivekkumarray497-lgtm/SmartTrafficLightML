import pandas as pd
import os

INPUT_FILE = "dataset/smart_traffic_management_dataset.csv"
OUTPUT_FILE = "dataset/cleaned_traffic_data.csv"

if not os.path.exists(INPUT_FILE):
    print("Dataset file not found!")
    print(f"Expected file: {INPUT_FILE}")
    exit()

print("=" * 60)
print("SMART TRAFFIC LIGHT ML - DATA CLEANING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE, sep="\t")

print("\nDataset loaded successfully!")
print(f"Original rows    : {df.shape[0]}")
print(f"Original columns : {df.shape[1]}")

print("\nOriginal Columns:")
for column in df.columns:
    print("-", column)

df.columns = df.columns.str.strip()

duplicate_count = df.duplicated().sum()

print(f"\nDuplicate records found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicate records removed: {duplicate_count}")
else:
    print("No duplicate records found.")

text_columns = [
    "weather_condition",
    "signal_status"
]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].astype("string").str.strip()

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    invalid_timestamp = df["timestamp"].isna().sum()

    print(f"\nInvalid timestamps: {invalid_timestamp}")

    if invalid_timestamp > 0:
        df = df.dropna(subset=["timestamp"])
        print("Rows with invalid timestamps removed.")

numeric_columns = [
    "location_id",
    "traffic_volume",
    "avg_vehicle_speed",
    "vehicle_count_cars",
    "vehicle_count_trucks",
    "vehicle_count_bikes",
    "temperature",
    "humidity",
    "accident_reported"
]

print("\nChecking numeric columns...")

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

for column in numeric_columns:
    if column in df.columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            print(
                f"Missing values filled in {column} "
                f"using median: {median_value}"
            )

categorical_columns = [
    "weather_condition",
    "signal_status"
]

for column in categorical_columns:
    if column in df.columns:
        missing_count = df[column].isnull().sum()

        if missing_count > 0:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
            print(
                f"Missing values filled in {column} "
                f"using mode: {mode_value}"
            )

if "location_id" in df.columns:
    invalid_location = ~df["location_id"].isin([1, 2, 3, 4, 5])
    invalid_count = invalid_location.sum()

    print(f"\nInvalid location IDs: {invalid_count}")

    if invalid_count > 0:
        df.loc[
            invalid_location,
            "location_id"
        ] = df["location_id"].median()

        df["location_id"] = (
            df["location_id"]
            .round()
            .astype(int)
        )

if "traffic_volume" in df.columns:
    invalid_traffic = df["traffic_volume"] < 0
    invalid_count = invalid_traffic.sum()

    print(f"Invalid traffic volume values: {invalid_count}")

    if invalid_count > 0:
        df.loc[
            invalid_traffic,
            "traffic_volume"
        ] = df["traffic_volume"].median()

if "avg_vehicle_speed" in df.columns:
    invalid_speed = df["avg_vehicle_speed"] < 0
    invalid_count = invalid_speed.sum()

    print(f"Invalid vehicle speed values: {invalid_count}")

    if invalid_count > 0:
        df.loc[
            invalid_speed,
            "avg_vehicle_speed"
        ] = df["avg_vehicle_speed"].median()

vehicle_columns = [
    "vehicle_count_cars",
    "vehicle_count_trucks",
    "vehicle_count_bikes"
]

for column in vehicle_columns:
    if column in df.columns:
        invalid_vehicle = df[column] < 0
        invalid_count = invalid_vehicle.sum()

        print(
            f"Invalid values in {column}: "
            f"{invalid_count}"
        )

        if invalid_count > 0:
            df.loc[
                invalid_vehicle,
                column
            ] = df[column].median()

if "temperature" in df.columns:
    invalid_temperature = (
        (df["temperature"] < -50) |
        (df["temperature"] > 60)
    )

    invalid_count = invalid_temperature.sum()

    print(
        f"Invalid temperature values: "
        f"{invalid_count}"
    )

    if invalid_count > 0:
        df.loc[
            invalid_temperature,
            "temperature"
        ] = df["temperature"].median()

if "accident_reported" in df.columns:
    invalid_accident = ~df[
        "accident_reported"
    ].isin([0, 1])

    invalid_count = invalid_accident.sum()

    print(
        f"Invalid accident_reported values: "
        f"{invalid_count}"
    )

    if invalid_count > 0:
        df.loc[
            invalid_accident,
            "accident_reported"
        ] = 0

    df["accident_reported"] = (
        df["accident_reported"]
        .astype(int)
    )

valid_weather = [
    "Sunny",
    "Cloudy",
    "Rainy",
    "Foggy",
    "Windy"
]

if "weather_condition" in df.columns:
    invalid_weather = ~df[
        "weather_condition"
    ].isin(valid_weather)

    invalid_count = invalid_weather.sum()

    print(
        f"Invalid weather conditions: "
        f"{invalid_count}"
    )

    if invalid_count > 0:
        df.loc[
            invalid_weather,
            "weather_condition"
        ] = df["weather_condition"].mode()[0]

valid_signals = [
    "Red",
    "Yellow",
    "Green"
]

if "signal_status" in df.columns:
    invalid_signal = ~df[
        "signal_status"
    ].isin(valid_signals)

    invalid_count = invalid_signal.sum()

    print(
        f"Invalid signal statuses: "
        f"{invalid_count}"
    )

    if invalid_count > 0:
        df.loc[
            invalid_signal,
            "signal_status"
        ] = df["signal_status"].mode()[0]

if "timestamp" in df.columns:
    df = df.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

print(
    "\nDuplicates AFTER cleaning:",
    df.duplicated().sum()
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Original dataset rows : {pd.read_csv(INPUT_FILE).shape[0]}")
print(f"Cleaned dataset rows  : {df.shape[0]}")
print(f"Columns               : {df.shape[1]}")

print("\nCleaned file saved at:")
print(OUTPUT_FILE)

print("\nFinal dataset preview:")
print(df.head())

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)