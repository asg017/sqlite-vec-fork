SHELL = /bin/bash
.PHONY: all bench-10k bench-50k bench-100k bench-quick \
        setup base-db ext profile clean help

# ── Paths ──────────────────────────────────────────────────────
ROOT     = ..
EXT_SO   = $(ROOT)/dist/vec0.so
BASE_DB  = seed/base.db
UV_RUN   = uv run --managed-python --project $(ROOT)/tests
BENCH    = $(UV_RUN) python3 bench.py
PROFILE_CLI = $(ROOT)/dist/sqlite3-profile

# ── Default: quick smoke test at 10k ──────────────────────────
all: bench-10k

# ── Setup: download data + build extension ────────────────────
setup: ext base-db

ext: $(EXT_SO)

$(EXT_SO):
	$(MAKE) -C $(ROOT) loadable

base-db: $(BASE_DB)

$(BASE_DB):
	$(MAKE) -C seed all

# ── Benchmarks ────────────────────────────────────────────────
#
# Each target builds the extension and base.db if missing, then
# runs annoy configs (with int8 quantization) vs brute-force baselines.
# Results go to runs/<size>/results.db and are printed to stdout.

# Quick: 10k, just the best configs, fast turnaround (~1 min)
bench-quick: $(EXT_SO) $(BASE_DB)
	$(BENCH) --subset-size 10000 \
		--configs \
			"annoy-t50-q8:n_trees=50,quantizer=int8" \
			"brute-float:type=float" \
			"brute-bit:type=bit" \
		-k 10 -n 50 -o runs/quick

# 10k: full sweep of tree counts and quantizers (~3 min)
bench-10k: $(EXT_SO) $(BASE_DB)
	$(BENCH) --subset-size 10000 \
		--configs \
			"annoy-t10:n_trees=10" \
			"annoy-t25:n_trees=25" \
			"annoy-t50:n_trees=50" \
			"annoy-t50-q8:n_trees=50,quantizer=int8" \
			"annoy-t100:n_trees=100" \
			"annoy-t100-q8:n_trees=100,quantizer=int8" \
			"brute-float:type=float" \
			"brute-int8:type=int8" \
			"brute-bit:type=bit" \
		-k 10 -n 50 -o runs/10k

# 50k: medium scale (~10 min)
bench-50k: $(EXT_SO) $(BASE_DB)
	$(BENCH) --subset-size 50000 \
		--configs \
			"annoy-t25:n_trees=25" \
			"annoy-t50:n_trees=50" \
			"annoy-t50-q8:n_trees=50,quantizer=int8" \
			"annoy-t100:n_trees=100" \
			"annoy-t100-q8:n_trees=100,quantizer=int8" \
			"brute-float:type=float" \
			"brute-int8:type=int8" \
			"brute-bit:type=bit" \
		-k 10 -n 100 -o runs/50k

# 100k: large scale (~15 min)
bench-100k: $(EXT_SO) $(BASE_DB)
	$(BENCH) --subset-size 100000 \
		--configs \
			"annoy-t50-q8:n_trees=50,quantizer=int8" \
			"annoy-t100-q8:n_trees=100,quantizer=int8" \
			"brute-float:type=float" \
			"brute-int8:type=int8" \
			"brute-bit:type=bit" \
		-k 10 -n 100 -o runs/100k

# All sizes
bench-all: bench-10k bench-50k bench-100k

# ── Profiling (requires perf) ─────────────────────────────────

$(PROFILE_CLI): $(ROOT)/sqlite-vec.c $(ROOT)/sqlite-vec-annoy.c
	$(CC) -g3 -O2 -mavx -DSQLITE_VEC_ENABLE_AVX \
		-DSQLITE_CORE -DSQLITE_EXTRA_INIT=core_init \
		-DSQLITE_THREADSAFE=0 \
		-o $@ \
		$(ROOT)/examples/sqlite3-cli/core_init.c \
		$(ROOT)/sqlite-vec.c $(ROOT)/vendor/sqlite3.c $(ROOT)/vendor/shell.c \
		-I$(ROOT) -I$(ROOT)/vendor -lm -lpthread -ldl

profile-query-10k: $(PROFILE_CLI) $(BASE_DB)
	$(PROFILE_CLI) < profile_query_only.sql

profile-perf-10k: $(PROFILE_CLI) $(BASE_DB)
	perf record -g --call-graph dwarf -F 999 \
		-o perf-query-10k.data \
		$(PROFILE_CLI) < profile_query_only.sql
	@echo "Run: perf report -i perf-query-10k.data --stdio --no-children --sort=symbol --percent-limit=0.5"

# ── Cleanup ───────────────────────────────────────────────────

clean:
	rm -rf runs/ *.data

clean-all: clean
	rm -f $(PROFILE_CLI)
	$(MAKE) -C seed clean

# ── Help ──────────────────────────────────────────────────────

help:
	@echo "Annoy Benchmark Suite"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Download COHERE dataset + build vec0 extension"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench-quick    Quick 10k smoke test (~1 min)"
	@echo "  make bench-10k      Full 10k sweep (~3 min)"
	@echo "  make bench-50k      50k sweep (~10 min)"
	@echo "  make bench-100k     100k sweep (~15 min)"
	@echo "  make bench-all      Run 10k + 50k + 100k"
	@echo ""
	@echo "Profiling:"
	@echo "  make profile-query-10k   Time KNN queries (10k, 200 queries)"
	@echo "  make profile-perf-10k    perf record KNN queries for flamegraph"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove benchmark outputs"
	@echo "  make clean-all       Remove outputs + base.db + profile CLI"
	@echo ""
	@echo "Config format for bench.py: name:key=val,key=val"
	@echo "  Annoy keys:    n_trees, search_k, quantizer (none|int8|binary)"
	@echo "  Baseline keys: type (float|int8|bit), oversample"
	@echo ""
	@echo "Example:"
	@echo "  make bench-quick                           # fast smoke test"
	@echo "  $(BENCH) --subset-size 25000 \\"
	@echo "    --configs 'my-cfg:n_trees=75,quantizer=int8' 'brute-float:type=float' \\"
	@echo "    -k 10 -n 50 -o runs/custom"
