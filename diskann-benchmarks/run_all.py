#!/usr/bin/env python3
"""Orchestrator: ground truth -> build -> bench for all sizes and configs."""
import argparse
import os
import subprocess
import sys

from build import ALL_CONFIGS, DISKANN_CONFIGS, BASELINE_CONFIGS

# At 1M, only run these configs by default (parameter sweeps are too slow)
LARGE_DATASET_CONFIGS = [
    "diskann-binary",
    "diskann-int8",
    "baseline-float",
    "baseline-bq",
    "baseline-int8",
]

LARGE_THRESHOLD = 500_000


def run_cmd(args_list):
    """Run a subprocess, streaming output."""
    print(f"  $ {' '.join(args_list)}", flush=True)
    result = subprocess.run(args_list)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all DiskANN benchmarks")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50000, 100000],
                        help="subset sizes to benchmark")
    parser.add_argument("-k", type=int, nargs="+", default=[10, 20, 50, 100],
                        help="k values for KNN")
    parser.add_argument("-n", type=int, default=100, help="number of query vectors")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="specific configs (default: all for small, subset for large)")
    parser.add_argument("-o", "--out-dir", default="runs/run1", help="output directory")
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--base-db", default=os.path.join(_script_dir, "seed", "base.db"))
    parser.add_argument("--ext", default=os.path.join(_script_dir, "..", "dist", "vec0"))
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="skip ground truth and build steps if files exist")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    max_k = max(args.k)

    for subset_size in args.sizes:
        print(f"\n{'='*60}")
        print(f"Subset size: {subset_size:,}")
        print(f"{'='*60}")

        # Determine configs for this size
        if args.configs:
            configs = args.configs
        elif subset_size >= LARGE_THRESHOLD:
            configs = LARGE_DATASET_CONFIGS
        else:
            configs = ALL_CONFIGS

        # Step 1: Ground truth
        gt_path = os.path.join(args.out_dir, f"ground_truth.{subset_size}.db")
        if args.skip_if_exists and os.path.exists(gt_path):
            print(f"\n[GT] Skipping (exists: {gt_path})")
        else:
            print(f"\n[GT] Generating ground truth for {subset_size:,} vectors...")
            ok = run_cmd([
                sys.executable, os.path.join(script_dir, "gen_ground_truth.py"),
                "--subset-size", str(subset_size),
                "-n", str(args.n),
                "-k", str(max_k),
                "--base-db", args.base_db,
                "--ext", args.ext,
                "-o", args.out_dir,
            ])
            if not ok:
                print(f"Ground truth generation failed for size {subset_size}, skipping")
                continue

        # Step 2: Build indexes
        print(f"\n[BUILD] Building {len(configs)} configs...")
        build_args = [
            sys.executable, os.path.join(script_dir, "build.py"),
            "--subset-size", str(subset_size),
            "--configs", *configs,
            "--base-db", args.base_db,
            "--ext", args.ext,
            "-o", args.out_dir,
        ]
        if args.skip_if_exists:
            build_args.append("--skip-if-exists")
        ok = run_cmd(build_args)
        if not ok:
            print(f"Build failed for size {subset_size}, skipping benchmarks")
            continue

        # Step 3: Benchmark all configs x k values
        print(f"\n[BENCH] Running benchmarks...")
        for config_name in configs:
            for k_val in args.k:
                db_path = os.path.join(args.out_dir, f"{config_name}.{subset_size}.db")
                if not os.path.exists(db_path):
                    print(f"  Skipping {config_name} k={k_val} (db not found)")
                    continue

                print(f"\n  --- {config_name} k={k_val} ---")
                run_cmd([
                    sys.executable, os.path.join(script_dir, "bench.py"),
                    "--config", config_name,
                    "--subset-size", str(subset_size),
                    "-k", str(k_val),
                    "-n", str(args.n),
                    "--base-db", args.base_db,
                    "--ext", args.ext,
                    "-o", args.out_dir,
                ])

    # Final summary
    results_path = os.path.join(args.out_dir, "results.db")
    if os.path.exists(results_path):
        import sqlite3
        db = sqlite3.connect(results_path)
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")

        rows = db.execute(
            "SELECT config_name, subset_size, k, mean_ms, recall "
            "FROM bench_results ORDER BY subset_size, k, config_name"
        ).fetchall()

        if rows:
            print(f"\n{'config':>25} {'size':>8} {'k':>4} {'mean_ms':>10} {'recall':>8}")
            print("-" * 60)
            for r in rows:
                print(f"{r[0]:>25} {r[1]:>8} {r[2]:>4} {r[3]:>10.3f} {r[4]:>8.4f}")

        db.close()

    print(f"\nDone! Results in {results_path}")


if __name__ == "__main__":
    main()
