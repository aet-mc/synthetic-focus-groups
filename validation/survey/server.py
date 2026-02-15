"""Minimal survey collection server. Stores responses as JSON files."""
import json
import os
import uuid
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "human_responses"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SURVEY_DIR = Path(__file__).parent


class SurveyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SURVEY_DIR), **kwargs)

    def do_POST(self):
        if self.path == "/api/submit":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                data["_id"] = str(uuid.uuid4())
                data["_server_ts"] = datetime.now(timezone.utc).isoformat()
                data["_ip"] = self.client_address[0]

                fname = DATA_DIR / f"{data['_id']}.json"
                fname.write_text(json.dumps(data, indent=2))

                # Update count
                count_file = DATA_DIR / "_count.txt"
                count = int(count_file.read_text().strip()) if count_file.exists() else 0
                count += 1
                count_file.write_text(str(count))

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "id": data["_id"], "count": count}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress GET logs, show POST logs
        if "POST" in str(args):
            super().log_message(format, *args)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    server = HTTPServer(("0.0.0.0", port), SurveyHandler)
    print(f"Survey server running on http://0.0.0.0:{port}")
    print(f"Responses saved to {DATA_DIR}")
    server.serve_forever()
