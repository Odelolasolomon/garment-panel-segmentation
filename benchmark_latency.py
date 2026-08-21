"""
CPU inference latency benchmark.

Satisfies the assessment's constraint: "Inference must run on CPU. Report
per-image latency on your machine and state the hardware."

Usage:
    python benchmark_latency.py --weights weights/best.pt --image sample.jpg --runs 50

Run this on the SAME machine (or Kaggle CPU-only session) you report
numbers from in the README, and paste this script's own hardware-info
output directly into the README rather than re-typing it -- keeps the
reported numbers and the reported hardware honestly paired.
"""
import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from predict import load_model, predict_mask


def print_hardware_info():
    print("=== Hardware ===")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU threads available to torch: {torch.get_num_threads()}")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    print(f"CPU model: {line.split(':', 1)[1].strip()}")
                    break
        with open("/proc/cpuinfo") as f:
            core_count = sum(1 for line in f if line.startswith("processor"))
        print(f"Logical cores: {core_count}")
    except FileNotFoundError:
        print("(/proc/cpuinfo not available on this platform)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    print_hardware_info()

    device = torch.device("cpu")
    torch.set_num_threads(torch.get_num_threads())  # explicit, reported above

    model = load_model(args.weights, device)

    for _ in range(args.warmup):
        predict_mask(model, args.image, device)

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        predict_mask(model, args.image, device)
        times.append((time.perf_counter() - t0) * 1000)

    print("=== Latency (forward pass + postprocessing, per image) ===")
    print(f"Runs: {args.runs}")
    print(f"Mean:   {statistics.mean(times):.1f} ms")
    print(f"Median: {statistics.median(times):.1f} ms")
    print(f"P95:    {sorted(times)[int(0.95 * len(times))]:.1f} ms")
    print(f"Min:    {min(times):.1f} ms")
    print(f"Max:    {max(times):.1f} ms")


if __name__ == "__main__":
    main()
