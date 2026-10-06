"""Benchmark suite: starts the server in different configurations, drives it
with the C++ load generator, and writes JSON results to bench/results/.
Plot them with bench/plot_results.py.

Experiments (GPU unless noted):
  concurrency  closed loop, 1..256 connections, batching off vs on
  open_loop    open loop (Poisson arrivals), offered load sweep, off vs on
  max_delay    open loop at a fixed rate, sweeping --max-delay-us
  cpu          closed loop on CPU, off vs on (shows batching does not help there)

Usage (from the repo root, inside the venv, after building):
  python bench/run_benchmarks.py --build ~/builds/cpp_ml_inference_server/cuda [--only concurrency,open_loop]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
PORT = 18100

# Batching off = same pipeline with one request per run; extra workers so the
# GPU still gets concurrent work. Batching on = max 16 per batch, 2 ms delay.
OFF = ["--max-batch", "1", "--workers", "4"]
ON = ["--max-batch", "16", "--workers", "2", "--max-delay-us", "2000"]


class Server:
    def __init__(self, build: Path, device: str, args: list[str]):
        self.cmd = [str(build / "inference_server"), "--model", str(ROOT / "models" / "resnet18.onnx"),
                    "--device", device, "--port", str(PORT), *args]
        self.proc = None

    def __enter__(self):
        self.log = open(RESULTS / "server.log", "a")
        self.log.write(f"\n=== {' '.join(self.cmd)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early: {self.cmd}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1).read()
                return self
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("server did not become healthy")

    def __exit__(self, *exc):
        self.proc.send_signal(signal.SIGTERM)
        rc = self.proc.wait(timeout=60)
        self.log.close()
        if rc != 0:
            print(f"  warning: server exited with {rc}", file=sys.stderr)


def loadgen(build: Path, mode: str, connections: int, rate: float = 0, duration: float = 5, warmup: float = 1) -> dict:
    cmd = [str(build / "loadgen"), "--port", str(PORT), "--mode", mode, "--connections", str(connections),
           "--duration", str(duration), "--warmup", str(warmup), "--json",
           "--input", str(ROOT / "models" / "ref_input.bin")]
    if mode == "open":
        cmd += ["--rate", str(rate)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    line = out.stdout.strip().splitlines()[-1]
    return json.loads(line)


def save(name: str, rows: list[dict]) -> None:
    (RESULTS / f"{name}.json").write_text(json.dumps(rows, indent=1))
    print(f"  -> bench/results/{name}.json ({len(rows)} rows)")


def show(tag: str, r: dict) -> None:
    print(f"  {tag:<28} {r['throughput_rps']:8.1f} req/s  p50 {r['p50_ms']:7.2f}  p99 {r['p99_ms']:8.2f} ms  "
          f"batch {r['avg_batch']:5.2f}  errors {r['errors']}", flush=True)


def exp_concurrency(build: Path) -> None:
    print("== concurrency sweep (closed loop, GPU)")
    rows = []
    for label, args in [("off", OFF), ("on", ON)]:
        with Server(build, "cuda", args):
            for c in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                r = loadgen(build, "closed", c)
                r["batching"] = label
                rows.append(r)
                show(f"{label} c={c}", r)
    save("concurrency", rows)


def exp_open_loop(build: Path) -> None:
    print("== offered-load sweep (open loop, GPU)")
    rows = []
    plans = [("off", OFF, [100, 200, 300, 400, 500, 600, 700, 800]),
             ("on", ON, [100, 250, 500, 750, 1000, 1250, 1500, 1750, 1900, 2000, 2100])]
    for label, args, rates in plans:
        with Server(build, "cuda", args):
            for rate in rates:
                r = loadgen(build, "open", 256, rate=rate)
                r["batching"] = label
                rows.append(r)
                show(f"{label} rate={rate}", r)
    save("open_loop", rows)


def exp_max_delay(build: Path) -> None:
    print("== max-delay sweep (open loop 1000 req/s, GPU, batching on)")
    rows = []
    for delay in [0, 250, 500, 1000, 2000, 4000, 8000]:
        args = ["--max-batch", "16", "--workers", "2", "--max-delay-us", str(delay)]
        with Server(build, "cuda", args):
            r = loadgen(build, "open", 256, rate=1000, duration=10)
            r["max_delay_us"] = delay
            rows.append(r)
            show(f"delay={delay}us", r)
    save("max_delay", rows)


def exp_cpu(build: Path) -> None:
    print("== CPU (closed loop, 32 connections)")
    rows = []
    for label, args in [("off", ["--max-batch", "1", "--workers", "4", "--ort-threads", "4"]),
                        ("on", ["--max-batch", "8", "--workers", "4", "--ort-threads", "4"])]:
        with Server(build, "cpu", args):
            r = loadgen(build, "closed", 32, duration=8, warmup=2)
            r["batching"] = label
            rows.append(r)
            show(f"cpu {label}", r)
    save("cpu", rows)


EXPERIMENTS = {"concurrency": exp_concurrency, "open_loop": exp_open_loop, "max_delay": exp_max_delay,
               "cpu": exp_cpu}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True, help="CMake build dir with inference_server and loadgen")
    ap.add_argument("--only", default=",".join(EXPERIMENTS), help="comma-separated subset")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    for name in args.only.split(","):
        EXPERIMENTS[name](args.build.expanduser())


if __name__ == "__main__":
    main()
