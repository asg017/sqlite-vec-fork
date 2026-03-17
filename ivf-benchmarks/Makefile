ROOT := $(shell cd .. && pwd)
CLI := $(ROOT)/dist/sqlite3
BASE_DB := $(ROOT)/benchmark2/zilliz/seed/base.db
RUNS := runs

$(RUNS):
	mkdir -p $(RUNS)

# ============================================================================
# Build CLI if needed
# ============================================================================

$(CLI):
	$(MAKE) -C .. cli

# ============================================================================
# Raw timing profile — saves .timer output from profile_ivf.sql
# ============================================================================

profile-timing: $(CLI) $(RUNS)
	cd $(ROOT) && $(CLI) < ivf-benchmarks/profile_ivf.sql > ivf-benchmarks/$(RUNS)/profile_ivf_raw.txt 2>&1
	@echo "Saved: $(RUNS)/profile_ivf_raw.txt"
	@grep -E "^(===|Run Time)" $(RUNS)/profile_ivf_raw.txt

# ============================================================================
# CPU profile with Instruments (Time Profiler)
# Produces .trace file — open in Instruments.app for flamegraph/hotspots
# ============================================================================

PROF_DB := /tmp/ivf_profile_workload.db

# Set up a trained IVF table for profiling
profile-setup: $(CLI)
	@rm -f $(PROF_DB)
	$(CLI) $(PROF_DB) "\
		ATTACH DATABASE '$(BASE_DB)' AS base; \
		PRAGMA page_size=8192; \
		CREATE VIRTUAL TABLE vec_items USING vec0( \
			id integer primary key, \
			embedding float[768] distance_metric=cosine \
			indexed by ivf(nlist=16, nprobe=4) \
		); \
		INSERT INTO vec_items(id, embedding) \
			SELECT id, vector FROM base.train WHERE id < 316; \
		INSERT INTO vec_items(id) VALUES ('compute-centroids');"
	@echo "Profile DB ready: $(PROF_DB)"

# Profile 30k trained inserts
profile-insert: profile-setup $(RUNS)
	xcrun xctrace record \
		--template 'Time Profiler' \
		--output $(RUNS)/profile_insert.trace \
		--launch -- $(CLI) $(PROF_DB) "\
			ATTACH DATABASE '$(BASE_DB)' AS base; \
			INSERT INTO vec_items(id, embedding) \
				SELECT id, vector FROM base.train WHERE id >= 316 AND id < 30316;"
	@echo "Open with: open $(RUNS)/profile_insert.trace"

# Profile KNN queries on 30k vectors
profile-query: $(RUNS)
	@test -f $(PROF_DB) || (echo "Run 'make profile-insert' first" && exit 1)
	xcrun xctrace record \
		--template 'Time Profiler' \
		--output $(RUNS)/profile_query.trace \
		--launch -- $(CLI) $(PROF_DB) "\
			ATTACH DATABASE '$(BASE_DB)' AS base; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=0) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=1) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=2) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=3) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=4) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=5) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=6) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=7) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=8) AND k=10; \
			SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=9) AND k=10;"
	@echo "Open with: open $(RUNS)/profile_query.trace"

# Profile both insert and query
profile: profile-insert profile-query

# ============================================================================
# macOS sample tool — lighter weight, text output
# ============================================================================

profile-timing-1m: $(CLI) $(RUNS)
	cd $(ROOT) && $(CLI) < ivf-benchmarks/profile_insert_1m.sql > ivf-benchmarks/$(RUNS)/profile_insert_1m_raw.txt 2>&1
	@echo "Saved: $(RUNS)/profile_insert_1m_raw.txt"
	@grep -E "^(===|Run Time)" $(RUNS)/profile_insert_1m_raw.txt

profile-sample-insert-1m: $(CLI) $(RUNS)
	@rm -f /tmp/ivf_prof_1m.db
	$(CLI) /tmp/ivf_prof_1m.db "\
		ATTACH DATABASE '$(BASE_DB)' AS base; \
		PRAGMA page_size=8192; \
		CREATE VIRTUAL TABLE vec_items USING vec0( \
			id integer primary key, \
			embedding float[768] distance_metric=cosine \
			indexed by ivf(nlist=128, nprobe=16) \
		); \
		INSERT INTO vec_items(id, embedding) \
			SELECT id, vector FROM base.train WHERE id < 8192; \
		INSERT INTO vec_items(id) VALUES ('compute-centroids'); \
		INSERT INTO vec_items(id, embedding) \
			SELECT id, vector FROM base.train WHERE id >= 8192 AND id < 18192;"
	@echo "Setup: 18k vectors trained. Profiling 30k inserts..."
	$(CLI) /tmp/ivf_prof_1m.db "\
		ATTACH DATABASE '$(BASE_DB)' AS base; \
		INSERT INTO vec_items(id, embedding) \
			SELECT id, vector FROM base.train WHERE id >= 18192 AND id < 48192;" &\
	PID=$$!; sleep 0.5; \
	sample $$PID 5 -file $(RUNS)/sample_insert_1m.txt; \
	wait $$PID 2>/dev/null; \
	echo "Saved: $(RUNS)/sample_insert_1m.txt"

profile-sample-insert: profile-setup $(RUNS)
	$(CLI) $(PROF_DB) "\
		ATTACH DATABASE '$(BASE_DB)' AS base; \
		INSERT INTO vec_items(id, embedding) \
			SELECT id, vector FROM base.train WHERE id >= 316 AND id < 10316;" &\
	PID=$$!; sleep 0.3; \
	sample $$PID 3 -file $(RUNS)/sample_insert.txt; \
	wait $$PID 2>/dev/null; \
	echo "Saved: $(RUNS)/sample_insert.txt"

profile-sample-query: $(CLI) $(RUNS)
	@# Build a 100k trained IVF table, then run 200 queries (slow enough to sample)
	@rm -f /tmp/ivf_profile_query.db
	$(CLI) /tmp/ivf_profile_query.db "\
		ATTACH DATABASE '$(BASE_DB)' AS base; \
		PRAGMA page_size=8192; \
		CREATE VIRTUAL TABLE vec_items USING vec0( \
			id integer primary key, \
			embedding float[768] distance_metric=cosine \
			indexed by ivf(nlist=128, nprobe=16) \
		); \
		INSERT INTO vec_items(id, embedding) \
			SELECT id, vector FROM base.train WHERE id < 1000; \
		INSERT INTO vec_items(id) VALUES ('compute-centroids'); \
		INSERT INTO vec_items(id, embedding) \
			SELECT id, vector FROM base.train WHERE id >= 1000 AND id < 100000;"
	@echo "Setup done (100k vectors). Running 200 queries..."
	python3 profile_query_workload.py /tmp/ivf_profile_query.db &\
	PID=$$!; sleep 0.5; \
	sample $$PID 5 -file $(RUNS)/sample_query.txt; \
	wait $$PID 2>/dev/null; \
	echo "Saved: $(RUNS)/sample_query.txt"

# ============================================================================
# Benchmarks
# ============================================================================

bench-100k: $(CLI)
	python3 bench_ivf.py --subset-size 100000 --nlist 128 --nprobe 16 -o $(RUNS)

bench-1m-sweep: $(CLI)
	python3 bench_ivf_1m.py | tee $(RUNS)/1m_sweep.log

bench-1m-baselines: $(CLI)
	python3 bench_baselines_1m.py | tee $(RUNS)/1m_baselines.log

# ============================================================================
# Parse results for analysis
# ============================================================================

parse-profile:
	python3 parse_profile.py $(RUNS)/profile_ivf_raw.txt

# Parse sample data into SQLite + print top functions
parse-sample-insert: $(RUNS)/sample_insert.txt
	python3 parse_sample.py $(RUNS)/sample_insert.txt --top 25

parse-sample-query: $(RUNS)/sample_query.txt
	python3 parse_sample.py $(RUNS)/sample_query.txt --top 25

parse-sample-insert-ivf:
	python3 parse_sample.py $(RUNS)/sample_insert.txt --filter ivf --top 20

parse-sample-query-ivf:
	python3 parse_sample.py $(RUNS)/sample_query.txt --filter ivf --top 20

# Full insert profiling workflow: build, run, sample, parse
profile-insert-full: profile-sample-insert parse-sample-insert

# Full query profiling workflow: build, run, sample, parse
profile-query-full: profile-sample-query parse-sample-query

summary:
	@echo ""
	@echo "=== IVF Sweep (1M) ==="
	@grep -A 20 "FINAL RESULTS" $(RUNS)/1m_sweep.log 2>/dev/null || echo "(not yet run — make bench-1m-sweep)"
	@echo ""
	@echo "=== Baselines (1M) ==="
	@grep -A 10 "BASELINE RESULTS" $(RUNS)/1m_baselines.log 2>/dev/null || echo "(not yet run — make bench-1m-baselines)"

clean:
	rm -rf $(RUNS)/*.trace $(RUNS)/*.txt $(RUNS)/*.log $(RUNS)/*.db /tmp/ivf_profile_workload.db

.PHONY: profile-timing profile-setup profile-insert profile-query profile \
        profile-sample-insert profile-sample-query \
        profile-insert-full profile-query-full \
        parse-profile parse-sample-insert parse-sample-query \
        parse-sample-insert-ivf parse-sample-query-ivf \
        bench-100k bench-1m-sweep bench-1m-baselines summary clean
