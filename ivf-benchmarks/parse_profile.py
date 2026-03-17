#!/usr/bin/env python3
"""Parse profile_ivf_raw.txt and print a summary table.

Usage:
  python3 parse_profile.py [runs/profile_ivf_raw.txt]
"""
import re
import sys


def parse_profile(path):
    phases = []
    current_phase = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== ") and line.endswith(" ==="):
                current_phase = line[4:-4]
            elif line.startswith("Run Time:"):
                m = re.search(r"real ([\d.]+)", line)
                if m:
                    real_s = float(m.group(1))
                    phases.append((current_phase or "(setup)", real_s))
                    current_phase = None  # reset for next unnamed phase

    # Group phases
    setup_time = 0
    insert_phases = []
    query_times = []

    for name, t in phases:
        if name == "(setup)":
            setup_time += t
        elif "Insert" in name or "insert" in name:
            insert_phases.append((name, t))
        elif name == "Queries" or name is None:
            query_times.append(t)
        elif "Train" in name or "train" in name:
            insert_phases.append((name, t))
        else:
            query_times.append(t)

    # For queries, the first "Queries" phase header is followed by individual query times
    # Re-parse to group properly
    query_times = []
    in_queries = False
    for name, t in phases:
        if name == "Queries":
            in_queries = True
            continue
        if in_queries:
            query_times.append(t)

    return insert_phases, query_times


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/profile_ivf_raw.txt"

    insert_phases, query_times = parse_profile(path)

    print(f"{'Phase':<45} {'Time (s)':>10} {'Rate':>12}")
    print("-" * 70)

    for name, t in insert_phases:
        # Try to extract vector count from phase name
        rate = ""
        m = re.search(r"(\d+) vectors", name)
        if m:
            n = int(m.group(1))
            rate = f"{n/t:.0f} vec/s" if t > 0 else ""
        # Also check "next NNNNN"
        m2 = re.search(r"next (\d+)", name)
        if m2:
            n = int(m2.group(1))
            rate = f"{n/t:.0f} vec/s" if t > 0 else ""
        print(f"  {name:<43} {t:>10.3f} {rate:>12}")

    total_insert = sum(t for _, t in insert_phases)
    print(f"  {'TOTAL INSERT':<43} {total_insert:>10.3f}")

    if query_times:
        print()
        print(f"  {'Queries':>43} {'each (ms)':>10}")
        print(f"  {'-'*43} {'-'*10}")
        for i, t in enumerate(query_times):
            print(f"  {'  query ' + str(i):<43} {t*1000:>10.1f}")
        mean_ms = sum(query_times) / len(query_times) * 1000
        median_ms = sorted(query_times)[len(query_times)//2] * 1000
        print(f"  {'  MEAN':<43} {mean_ms:>10.1f}")
        print(f"  {'  MEDIAN':<43} {median_ms:>10.1f}")


if __name__ == "__main__":
    main()
