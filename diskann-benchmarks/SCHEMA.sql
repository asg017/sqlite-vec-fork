CREATE TABLE IF NOT EXISTS build_results (
  config_name  TEXT NOT NULL,
  subset_size  INTEGER NOT NULL,
  db_path      TEXT NOT NULL,
  build_time_s REAL NOT NULL,
  rows         INTEGER NOT NULL,
  file_size_mb REAL NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (config_name, subset_size)
);

CREATE TABLE IF NOT EXISTS bench_results (
  config_name  TEXT NOT NULL,
  subset_size  INTEGER NOT NULL,
  k            INTEGER NOT NULL,
  n            INTEGER NOT NULL,
  mean_ms      REAL NOT NULL,
  median_ms    REAL NOT NULL,
  p99_ms       REAL NOT NULL,
  total_ms     REAL NOT NULL,
  qps          REAL NOT NULL,
  recall       REAL NOT NULL,
  db_path      TEXT NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (config_name, subset_size, k)
);

CREATE TABLE IF NOT EXISTS insert_throughput (
  strategy    TEXT NOT NULL,
  subset_size INTEGER NOT NULL,
  checkpoint  INTEGER NOT NULL,
  elapsed_s   REAL NOT NULL,
  rows_per_s  REAL NOT NULL,
  PRIMARY KEY (strategy, subset_size, checkpoint)
);

CREATE TABLE IF NOT EXISTS insert_summary (
  strategy    TEXT NOT NULL,
  subset_size INTEGER NOT NULL,
  total_time_s REAL NOT NULL,
  total_rows  INTEGER NOT NULL,
  avg_rows_per_s REAL NOT NULL,
  recall_at_10 REAL,
  file_size_mb REAL NOT NULL,
  PRIMARY KEY (strategy, subset_size)
);
