"""Quick end-to-end check of a running inference server.

1. Correctness: sends each sample of models/ref_input.bin and checks the
   server's top-1 class against the PyTorch reference output.
2. Load: N concurrent keep-alive clients for a few seconds; prints throughput
   and latency percentiles. (The proper load generator comes in Phase 6.)

Usage: python scripts/infer_client.py [--port 8080] [--clients 16] [--seconds 5]
"""

import argparse
import http.client
import json
import threading
import time
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
SAMPLE_SHAPE = (3, 224, 224)
NUM_CLASSES = 1000


def infer(conn: http.client.HTTPConnection, sample: np.ndarray) -> dict:
    conn.request("POST", "/v1/infer", body=sample.tobytes(),
                 headers={"Content-Type": "application/octet-stream"})
    resp = conn.getresponse()
    body = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {body!r}")
    return json.loads(body)


def check_correctness(port: int) -> None:
    x = np.fromfile(MODELS / "ref_input.bin", dtype=np.float32).reshape(-1, *SAMPLE_SHAPE)
    y = np.fromfile(MODELS / "ref_output.bin", dtype=np.float32).reshape(-1, NUM_CLASSES)
    conn = http.client.HTTPConnection("127.0.0.1", port)
    matches = 0
    for i in range(len(x)):
        got = infer(conn, x[i])["top_k"][0]["class"]
        matches += int(got == int(y[i].argmax()))
    conn.close()
    print(f"correctness: top-1 matches PyTorch for {matches}/{len(x)} samples")
    if matches != len(x):
        raise SystemExit(1)


def load_test(port: int, clients: int, seconds: float) -> None:
    sample = np.random.default_rng(0).standard_normal(SAMPLE_SHAPE, dtype=np.float32)
    latencies: list[float] = []
    lock = threading.Lock()
    deadline = time.perf_counter() + seconds

    def worker() -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port)
        local = []
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            infer(conn, sample)
            local.append(time.perf_counter() - t0)
        conn.close()
        with lock:
            latencies.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(clients)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start

    lat_ms = np.array(latencies) * 1000
    print(f"load: {clients} clients, {len(lat_ms)} requests in {elapsed:.1f}s "
          f"-> {len(lat_ms) / elapsed:.1f} req/s | latency ms "
          f"p50={np.percentile(lat_ms, 50):.1f} p90={np.percentile(lat_ms, 90):.1f} "
          f"p99={np.percentile(lat_ms, 99):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--clients", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    check_correctness(args.port)
    load_test(args.port, args.clients, args.seconds)


if __name__ == "__main__":
    main()
