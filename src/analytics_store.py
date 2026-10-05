import csv, os
from datetime import datetime
class AnalyticsStore:
    fields=['timestamp','cars','bikes','buses','trucks','total_vehicles','pedestrians','avg_speed','density','queue_length','avg_waiting_time','direction','signal']
    def __init__(self,path):
        self.path=path; os.makedirs(os.path.dirname(path),exist_ok=True)
        if not os.path.exists(path):
            with open(path,'w',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=self.fields).writeheader()
    def append(self,m,direction,signal):
        row={k:m.get(k,'') for k in self.fields if k in m}; row.update(timestamp=datetime.now().isoformat(timespec='seconds'),direction=direction,signal=signal)
        with open(self.path,'a',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=self.fields).writerow(row)
    def recent(self,n=60):
        try:
            with open(self.path,encoding='utf-8') as f: rows=list(csv.DictReader(f))[-n:]
            return rows
        except Exception:return []
