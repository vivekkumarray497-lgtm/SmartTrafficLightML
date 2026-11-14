import csv
import os
from datetime import datetime


class AnalyticsStore:
    fields = [
        "timestamp", "cars", "bikes", "buses", "trucks", "auto_rickshaws",
        "total_vehicles", "pedestrians", "avg_speed", "density", "queue_length",
        "avg_waiting_time", "direction", "signal",
    ]

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._write_rows([])
        else:
            self._migrate_header_if_needed()

    def _write_rows(self, rows):
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def _migrate_header_if_needed(self):
        try:
            with open(self.path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                old_fields = reader.fieldnames or []
                rows = list(reader)
            if old_fields != self.fields:
                self._write_rows(rows)
        except (OSError, csv.Error):
            self._write_rows([])

    def append(self, metrics, direction, signal):
        row = {key: metrics.get(key, "") for key in self.fields if key in metrics}
        row.update(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            direction=direction,
            signal=signal,
        )
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)

    def recent(self, n=60):
        try:
            with open(self.path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[-n:]
            return rows
        except (OSError, csv.Error):
            return []
