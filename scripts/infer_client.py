"""Quick end-to-end check of a running inference server.

1. Correctness: sends each sample of models/ref_input.bin and checks the
   server's top-1 class against the PyTorch reference output.
2. Load: N concurrent keep-alive clients for a few seconds, spread over
   several processes so the GIL doesn't cap the request rate. Prints
   throughput, latency percentiles and the average batch size the server
   reported. (The proper C++ load generator comes in Phase 6.)

Usage: python scripts/infer_client.py [--port 8080] [--clients 16]
                                      [--processes 4] [--seconds 5]
"""

import argparse
import http.client
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
SAMPLE_SHAPE = (3, 224, 224)
NUM_CLASSES = 1000


def infer(conn: http.client.HTTPConnection, body: bytes) -> dict:
    conn.request("POST", "/v1/infer", body=body,
                 headers={"Content-Type": "application/octet-stream"})
    resp = conn.getresponse()
    data = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {data!r}")
    return json.loads(data)


def check_correctness(port: int) -> None:
    x = np.fromfile(MODELS / "ref_input.bin", dtype=np.float32).reshape(-1, *SAMPLE_SHAPE)
    y = np.fromfile(MODELS / "ref_output.bin", dtype=np.float32).reshape(-1, NUM_CLASSES)
    conn = http.client.HTTPConnection("127.0.0.1", port)
    matches = 0
    for i in range(len(x)):
        got = infer(conn, x[i].tobytes())["top_k"][0]["class"]
        matches += int(got == int(y[i].argmax()))
    conn.close()
    print(f"correctness: top-1 matches PyTorch for {matches}/{len(x)} samples")
    if matches != len(x):
        raise SystemExit(1)


def run_process(args: tuple[int, int, float]) -> tuple[list[float], list[int], int]:
    """One process: `threads` keep-alive clients until the shared deadline.

    Returns (latencies in seconds, batch sizes reported, error count).
    """
    port, threads, deadline = args
    body = np.random.default_rng(0).standard_normal(SAMPLE_SHAPE, dtype=np.float32).tobytes()
    lat: list[float] = []
    batches: list[int] = []
    errors = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal errors
        conn = http.client.HTTPConnection("127.0.0.1", port)
        l_lat, l_batch, l_err = [], [], 0
        while time.monotonic() < deadline:  # CLOCK_MONOTONIC is shared across processes
            t0 = time.perf_counter()
            try:
                r = infer(conn, body)
            except Exception:
                l_err += 1
                conn.close()
                conn = http.client.HTTPConnection("127.0.0.1", port)
                continue
            l_lat.append(time.perf_counter() - t0)
            l_batch.append(r.get("batch_size", 1))
        conn.close()
        with lock:
            lat.extend(l_lat)
            batches.extend(l_batch)
            errors += l_err

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return lat, batches, errors


def load_test(port: int, clients: int, processes: int, seconds: float) -> None:
    processes = max(1, min(processes, clients))
    per_proc = [clients // processes + (1 if i < clients % processes else 0) for i in range(processes)]
    start = time.monotonic() + 0.5  # let the processes spin up first
    deadline = start + seconds
    with mp.Pool(processes) as pool:
        results = pool.map(run_process, [(port, n, deadline) for n in per_proc])

    lat = np.concatenate([np.array(r[0]) for r in results]) * 1000
    batches = np.concatenate([np.array(r[1]) for r in results])
    errors = sum(r[2] for r in results)
    print(f"load: {clients} clients x {processes} procs, {len(lat)} ok / {errors} errors in {seconds:.0f}s "
          f"-> {len(lat) / seconds:.1f} req/s | latency ms p50={np.percentile(lat, 50):.1f} "
          f"p90={np.percentile(lat, 90):.1f} p99={np.percentile(lat, 99):.1f} | "
          f"avg batch {batches.mean():.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--clients", type=int, default=16)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    if not args.skip_check:
        check_correctness(args.port)
    load_test(args.port, args.clients, args.processes, args.seconds)


if __name__ == "__main__":
    main()
