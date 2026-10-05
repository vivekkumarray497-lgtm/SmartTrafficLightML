import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))

from app import app

if __name__ == "__main__":
    host = os.environ.get("TRAFFIC_HOST", "127.0.0.1")
    port = int(os.environ.get("TRAFFIC_PORT", "5000"))
    print("=" * 60)
    print("SMART TRAFFIC LIGHT MANAGEMENT SYSTEM")
    print(f"Open: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
