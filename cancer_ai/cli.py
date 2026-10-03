"""
CLI Tool for Breast Cancer Sentinel 2.0.
"""

import sys
import time
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cancer_ai.config import settings
from cancer_ai.service import service
from cancer_ai.schemas import PatientFNAInput


def cmd_benchmark(args):
    print("=" * 70)
    print("🔬 BREAST CANCER SENTINEL 2.0 -- INFERENCE LATENCY BENCHMARK")
    print("=" * 70)
    sample = {
        "radius_mean": 17.99, "texture_mean": 21.60, "perimeter_mean": 122.8, "area_mean": 1001.0,
        "smoothness_mean": 0.118, "compactness_mean": 0.277, "concavity_mean": 0.300, "concave points_mean": 0.147,
        "symmetry_mean": 0.242, "fractal_dimension_mean": 0.078,
        "radius_se": 1.095, "texture_se": 0.905, "perimeter_se": 8.589, "area_se": 153.4,
        "smoothness_se": 0.006, "compactness_se": 0.049, "concavity_se": 0.053, "concave points_se": 0.015,
        "symmetry_se": 0.030, "fractal_dimension_se": 0.006,
        "radius_worst": 25.38, "texture_worst": 28.50, "perimeter_worst": 184.6, "area_worst": 2019.0,
        "smoothness_worst": 0.162, "compactness_worst": 0.665, "concavity_worst": 0.711, "concave points_worst": 0.265,
        "symmetry_worst": 0.460, "fractal_dimension_worst": 0.118
    }
    pt = PatientFNAInput(**sample)

    # Warmup
    for _ in range(5):
        service.evaluate_patient(pt)

    start = time.perf_counter()
    N = args.n
    for _ in range(N):
        service.evaluate_patient(pt)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / N) * 1000.0
    qps = N / elapsed
    print(f"Total Iterations: {N:,}")
    print(f"Average Latency: {avg_ms:.2f} ms per patient")
    print(f"Throughput: {qps:,.0f} evaluations/sec")
    print("=" * 70)


def cmd_serve(args):
    import uvicorn
    print(f"Starting Breast Cancer Sentinel API on {args.host}:{args.port}...")
    uvicorn.run("cancer_ai.api.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_dashboard(args):
    import subprocess
    dash = PROJECT_ROOT / "cancer_ai" / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dash), "--server.port", str(args.port)])


def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Sentinel CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("-n", type=int, default=100)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    p_dash = sub.add_parser("dashboard")
    p_dash.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()
    if args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)


if __name__ == "__main__":
    main()
