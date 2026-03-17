#!/usr/bin/env python3
"""Parse macOS `sample` output into structured data.

Extracts the call tree into a SQLite database for easy querying.

Usage:
  python3 parse_sample.py runs/sample_insert.txt [-o runs/sample_insert.db]
  python3 parse_sample.py runs/sample_query.txt --top 20
  python3 parse_sample.py runs/sample_insert.txt --filter ivf
"""
import argparse
import re
import sqlite3
import sys
import os


def parse_sample_file(path):
    """Parse a macOS sample output file into a list of (samples, depth, func, source) tuples."""
    entries = []
    in_call_graph = False

    with open(path) as f:
        for line in f:
            if "Call graph:" in line:
                in_call_graph = True
                continue
            if not in_call_graph:
                continue
            if line.strip() == "" or line.startswith("Total number"):
                continue

            # Match lines like: "    1234 funcName  (in binary) + offset  [addr]  file.c:123"
            m = re.match(
                r"^[\s+|!:]*(\d+)\s+(\S+)\s+\(in ([^)]+)\)\s*\+\s*\d+\s*\[0x[0-9a-f]+\](?:\s+(\S+:\d+))?",
                line,
            )
            if not m:
                continue

            samples = int(m.group(1))
            func = m.group(2)
            binary = m.group(3)
            source = m.group(4) or ""

            # Compute depth from leading whitespace/tree chars
            stripped = line.lstrip(" +|!:")
            depth = len(line) - len(stripped)

            entries.append({
                "samples": samples,
                "depth": depth,
                "function": func,
                "binary": binary,
                "source": source,
                "file": source.split(":")[0] if ":" in source else "",
                "line": int(source.split(":")[1]) if ":" in source else 0,
            })

    return entries


def aggregate_functions(entries):
    """Aggregate samples by function name (self time approximation)."""
    # The sample tool reports inclusive time. For a rough self-time estimate,
    # we take the raw sample count at each call site.
    by_func = {}
    for e in entries:
        key = (e["function"], e["source"])
        if key not in by_func:
            by_func[key] = {"function": e["function"], "source": e["source"],
                            "binary": e["binary"], "max_samples": 0, "occurrences": 0}
        by_func[key]["max_samples"] = max(by_func[key]["max_samples"], e["samples"])
        by_func[key]["occurrences"] += 1
    return sorted(by_func.values(), key=lambda x: -x["max_samples"])


def save_to_db(entries, aggregated, db_path):
    """Save parsed data to SQLite for easy querying."""
    db = sqlite3.connect(db_path)
    db.executescript("""
        DROP TABLE IF EXISTS call_tree;
        DROP TABLE IF EXISTS functions;

        CREATE TABLE call_tree (
            id INTEGER PRIMARY KEY,
            samples INTEGER,
            depth INTEGER,
            function TEXT,
            binary_name TEXT,
            source TEXT,
            file TEXT,
            line INTEGER
        );

        CREATE TABLE functions (
            function TEXT,
            source TEXT,
            binary_name TEXT,
            max_samples INTEGER,
            occurrences INTEGER
        );
    """)

    db.executemany(
        "INSERT INTO call_tree (samples, depth, function, binary_name, source, file, line) "
        "VALUES (:samples, :depth, :function, :binary, :source, :file, :line)",
        entries,
    )
    db.executemany(
        "INSERT INTO functions (function, source, binary_name, max_samples, occurrences) "
        "VALUES (:function, :source, :binary, :max_samples, :occurrences)",
        aggregated,
    )
    db.commit()
    db.close()


def print_top(aggregated, n=20, filter_str=None):
    """Print top functions by sample count."""
    items = aggregated
    if filter_str:
        items = [a for a in items if filter_str.lower() in a["function"].lower()
                 or filter_str.lower() in a["source"].lower()]

    total = max(a["max_samples"] for a in aggregated) if aggregated else 1
    print(f"{'samples':>8} {'%':>6} {'function':<40} {'source'}")
    print("-" * 90)
    for a in items[:n]:
        pct = a["max_samples"] / total * 100
        print(f"{a['max_samples']:>8} {pct:>5.1f}% {a['function']:<40} {a['source']}")


def main():
    parser = argparse.ArgumentParser(description="Parse macOS sample output")
    parser.add_argument("input", help="Path to sample output .txt file")
    parser.add_argument("-o", "--output", help="Output SQLite DB path (default: <input>.db)")
    parser.add_argument("--top", type=int, default=20, help="Show top N functions")
    parser.add_argument("--filter", help="Filter functions by substring")
    args = parser.parse_args()

    entries = parse_sample_file(args.input)
    aggregated = aggregate_functions(entries)

    db_path = args.output or args.input.replace(".txt", ".db")
    save_to_db(entries, aggregated, db_path)
    print(f"Saved {len(entries)} call tree entries to {db_path}\n")

    print_top(aggregated, args.top, args.filter)

    # Print summary categories
    print(f"\n--- Category summary ---")
    categories = {
        "distance/compute": ["ivf_l2_dist", "ivf_nearest_centroid", "ivf_scan_cells_from_stmt",
                              "ivf_candidate_cmp", "_platform_memmove", "memcpy"],
        "disk I/O": ["pread", "pwrite", "unixRead", "unixWrite"],
        "btree/page": ["getPageNormal", "accessPayload", "copyPayload",
                        "btreeOverwriteOverflowCell", "btreeOverwriteContent",
                        "readDbPage", "pagerStress", "pager_write_pagelist"],
        "sqlite overhead": ["sqlite3_step", "sqlite3VdbeExec", "sqlite3_blob_open",
                             "sqlite3_blob_read", "sqlite3_blob_write",
                             "sqlite3_prepare_v2", "sqlite3_reset"],
        "vec0 insert": ["vec0Update", "vec0Update_Insert", "vec0Update_InsertWriteFinalStep",
                         "vec0_rowids_update_position", "vec0Update_InsertNextAvailableStep",
                         "vec0Update_InsertRowidStep"],
        "ivf insert": ["ivf_insert", "ivf_cell_insert", "ivf_cell_find_or_create"],
        "ivf query": ["ivf_query_knn", "ivf_scan_cells_from_stmt", "vec0Filter_knn"],
        "sorting": ["qsort", "ivf_candidate_cmp"],
    }

    total_samples = max(a["max_samples"] for a in aggregated) if aggregated else 1
    func_samples = {a["function"]: a["max_samples"] for a in aggregated}
    for cat, funcs in categories.items():
        total_cat = sum(func_samples.get(f, 0) for f in funcs)
        if total_cat > 0:
            pct = total_cat / total_samples * 100
            print(f"  {cat:<25} {total_cat:>8} samples ({pct:.1f}%)")


if __name__ == "__main__":
    main()
