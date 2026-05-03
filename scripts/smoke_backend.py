import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def pick_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fetch(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status, response.read().decode("utf-8")


def main():
    port = pick_port()
    env = os.environ.copy()
    env["NEUROVIEW_SMOKE_TEST"] = "1"

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO_ROOT,
        env=env,
    )

    try:
        deadline = time.time() + 30
        last_error = None
        while time.time() < deadline:
            try:
                status, health_body = fetch(f"http://127.0.0.1:{port}/health")
                if status == 200:
                    break
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)
        else:
            raise RuntimeError(f"Backend did not start in time: {last_error}")

        health = json.loads(health_body)
        assert health["status"] == "ok", f"Unexpected health payload: {health}"
        assert health["mode"] == "smoke-test", f"Unexpected mode payload: {health}"

        root_status, root_body = fetch(f"http://127.0.0.1:{port}/")
        assert root_status == 200, f"Root returned {root_status}"
        assert "NeuroView MRI" in root_body, "Desktop UI title missing from root page"

        print("Smoke test passed: backend health endpoint and desktop UI shell are available.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
