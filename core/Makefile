PYTEST = pytest
JUNIT_SERIAL = serial.xml
JUNIT_EXCLUSIVE = exclusive.xml
JUNIT_MERGED = merged.xml

.PHONY: install tests tests-full clean ray-clean ray-status start stop restart sweep-pi-swap

# Build/repair the project venv so it EXACTLY matches uv.lock — the
# single source of truth every Mac in the cluster syncs to. `--frozen`
# refuses to silently re-resolve: if pyproject.toml changed without a
# fresh `uv lock`, this errors instead of drifting. Run on every node.
# pyproject.toml + uv.lock live at the repo root (one level up).
install:
	cd .. && uv sync --frozen
	@echo "✅ venv synced to uv.lock. Run the app with: uv run python main.py"

# Two-phase strategy that works around Ray + parallel-pytest instability:
#   SERIAL phase — `serial` (timing/benchmark) + `exclusive` (state-isolation)
#     tests as plain `pytest -n 0`. They flake under xdist/parallelism (Ray
#     cluster init/teardown races) but are rock-solid single-process, and few.
#   PARALLEL phase — all correctness tests, sharded across process-per-file
#     `-n 0` runs (scripts/run_parallel_tests.py): the reliable way to
#     parallelize Ray tests (each shard is its own process + Ray cluster).
# Every phase runs even if an earlier one fails; exits non-zero if any did.
tests:
	@rm -f $(JUNIT_SERIAL) $(JUNIT_EXCLUSIVE) $(JUNIT_MERGED) shard*.xml; rc=0; \
	echo "→ serial (timing/benchmark) tests — plain -n 0"; \
	PYTHONPATH=. uv run pytest -n 0 -m "not exhaustive and serial and not slow" --junitxml=$(JUNIT_SERIAL) || rc=1; \
	echo "→ exclusive tests — plain -n 0"; \
	PYTHONPATH=. uv run pytest -n 0 -m "not exhaustive and exclusive and not slow" --junitxml=$(JUNIT_EXCLUSIVE) || rc=1; \
	echo "→ parallel correctness tests — process-per-file shards"; \
	uv run python scripts/run_parallel_tests.py || rc=1; \
	python merge_junitxml.py $(JUNIT_SERIAL) $(JUNIT_EXCLUSIVE) shard*.xml > $(JUNIT_MERGED) 2>/dev/null || true; \
	[ $$rc -eq 0 ] && echo "✅ all phases green — report: $(JUNIT_MERGED)" || echo "❌ a phase failed — report: $(JUNIT_MERGED)"; \
	exit $$rc

# Same, but also include tests marked `slow` (threaded to the runner via INCLUDE_SLOW).
tests-full:
	@rm -f $(JUNIT_SERIAL) $(JUNIT_EXCLUSIVE) $(JUNIT_MERGED) shard*.xml; rc=0; \
	PYTHONPATH=. uv run pytest -n 0 -m "not exhaustive and serial" --junitxml=$(JUNIT_SERIAL) || rc=1; \
	PYTHONPATH=. uv run pytest -n 0 -m "not exhaustive and exclusive" --junitxml=$(JUNIT_EXCLUSIVE) || rc=1; \
	INCLUDE_SLOW=1 uv run python scripts/run_parallel_tests.py || rc=1; \
	python merge_junitxml.py $(JUNIT_SERIAL) $(JUNIT_EXCLUSIVE) shard*.xml > $(JUNIT_MERGED) 2>/dev/null || true; \
	exit $$rc

clean:
	rm -f $(JUNIT_SERIAL) $(JUNIT_EXCLUSIVE) $(JUNIT_MERGED) shard*.xml

# Aggressive Ray reset — as close to a reboot as you can get without one.
# Use when tests hang or `ray.init()` blows up with
# `ActorDiedError: ContextActor` / `LocalRayletDiedError`. Idempotent.
ray-clean:
	@echo "→ ray stop --force"
	-ray stop --force >/dev/null 2>&1
	@echo "→ pkill ray subprocesses (raylet, gcs_server, plasma_store, default_worker, ray:: workers)"
	-pkill -9 -f "raylet" >/dev/null 2>&1 || true
	-pkill -9 -f "gcs_server" >/dev/null 2>&1 || true
	-pkill -9 -f "plasma_store" >/dev/null 2>&1 || true
	-pkill -9 -f "default_worker" >/dev/null 2>&1 || true
	-pkill -9 -f "ray::" >/dev/null 2>&1 || true
	-pkill -9 -f "ray/_private/runtime_env" >/dev/null 2>&1 || true
	-pkill -9 -f "log_monitor" >/dev/null 2>&1 || true
	-pkill -9 -f "dashboard" >/dev/null 2>&1 || true
	-pkill -9 -f "monitor.py" >/dev/null 2>&1 || true
	@echo "→ pkill stray pytest"
	-pkill -9 -f "pytest" >/dev/null 2>&1 || true
	@sleep 1
	@echo "→ rm /tmp/ray and /tmp/ray-gw* (per-session state, plasma + spill dirs, sockets)"
	-rm -rf /tmp/ray 2>/dev/null
	-rm -rf /tmp/ray-* 2>/dev/null
	-rm -rf /tmp/ray-gw* 2>/dev/null
	-rm -rf /tmp/ray_* 2>/dev/null
	@echo "→ rm /var/folders/.../T/ray* (macOS per-user temp ray state)"
	-find /var/folders -maxdepth 5 -name "ray*" -type d -exec rm -rf {} + 2>/dev/null || true
	-find /var/folders -maxdepth 5 -name "ray-*" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "→ rm /tmp/claude-*/tmp* (test temp dirs leaked by previous runs)"
	-find /tmp -maxdepth 3 -name "tmp*" -type d -mmin +5 -exec rm -rf {} + 2>/dev/null || true
	@echo "→ check no stragglers"
	@count=$$(pgrep -f "raylet|gcs_server|plasma_store|default_worker|ray::" 2>/dev/null | wc -l | tr -d ' '); \
	if [ "$$count" -gt 0 ]; then \
	  echo "⚠  $$count ray processes still alive after cleanup (may take a moment to exit)"; \
	  pgrep -fl "raylet|gcs_server|plasma_store|default_worker|ray::" 2>/dev/null | head -5; \
	else \
	  echo "✅ ray clean: no ray procs, no /tmp/ray dirs"; \
	fi
	@count=$$(ls /tmp/ray* 2>/dev/null | wc -l | tr -d ' '); \
	if [ "$$count" -gt 0 ]; then \
	  echo "⚠  /tmp/ray* paths remain:"; \
	  ls /tmp/ray* 2>/dev/null | head -5; \
	fi
	@netstat_count=$$(netstat -anp tcp 2>/dev/null | grep -cE "TIME_WAIT|CLOSE_WAIT" || echo 0); \
	if [ "$$netstat_count" -gt 1000 ]; then \
	  echo "⚠  $$netstat_count sockets in TIME_WAIT/CLOSE_WAIT — high churn; may take ~60s to drain"; \
	fi

# Quick status: what's alive and what's on disk?
ray-status:
	@echo "=== ray processes ==="
	@pgrep -fl "raylet|gcs_server|plasma_store|default_worker|ray::|dashboard|log_monitor" 2>/dev/null | head -10 || echo "  (none)"
	@echo "=== ray temp dirs ==="
	@ls -d /tmp/ray* 2>/dev/null | head -5 || echo "  (none)"
	@echo "=== socket churn ==="
	@printf "  TIME_WAIT/CLOSE_WAIT: %s\n" $$(netstat -anp tcp 2>/dev/null | grep -cE "TIME_WAIT|CLOSE_WAIT")
	@echo "=== swap usage ==="
	@sysctl vm.swapusage 2>/dev/null | sed 's/^/  /'

# --- App lifecycle ---------------------------------------------------------
# Start the live app (Ray head + gunicorn web server) in the FOREGROUND.
# Run from core/. Ctrl-C stops it (main.py's atexit cleans up gunicorn + ray).
start: sweep-pi-swap
	uv run python main.py

# Reclaim orphaned PixInsight swap files (~PI~*.swp) stranded by hard-killed PI
# sessions — the inactivity watchdog, `make stop`, or crashes. PI only deletes
# its swap on a GRACEFUL quit, and nothing else reclaims these: not an app
# restart, not a PI relaunch, not even a reboot (they live in the Caches dir,
# which macOS never clears). Left alone, these multi-GB gigapixel-mosaic swaps
# accumulated to 79 GB and wedged the boot volume. Safe ONLY when no PI is
# running — then every ~PI~*.swp is abandoned; if one is alive we skip the sweep
# so an in-use swap is never deleted, and the next clean start reclaims it.
sweep-pi-swap:
	@if pgrep -f 'PixInsight.app' >/dev/null 2>&1; then \
		echo "→ PixInsight running — skipping PI swap sweep"; \
	else \
		for d in "$$(getconf DARWIN_USER_CACHE_DIR)" "$$(getconf DARWIN_USER_TEMP_DIR)"; do \
			[ -d "$$d" ] || continue; \
			n=$$(find "$$d" -maxdepth 1 -name '~PI~*.swp' 2>/dev/null | wc -l | tr -d ' '); \
			[ "$$n" -eq 0 ] && continue; \
			sz=$$(find "$$d" -maxdepth 1 -name '~PI~*.swp' -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1); \
			find "$$d" -maxdepth 1 -name '~PI~*.swp' -delete 2>/dev/null; \
			echo "→ swept $$n orphaned PI swap file(s) ($$sz) from $$d"; \
		done; \
	fi

# Stop the live app: kill the DRIVER (main.py) first, then ray + gunicorn +
# PixInsight. `make stop` used to leave `main.py` running (it only stopped
# ray/gunicorn/PixInsight), so `make stop` + a fresh launch STACKED a second
# driver — multiple mains then fought over the same diskcache/output dir and the
# derivation crawled to a halt. Killing main.py here is the fix.
# NOTE the `[g]unicorn` / `[P]ixInsight` / ` [m]ain.py` bracket trick: `pkill -f`
# matches full command lines, and this recipe's own shell has the pattern in its
# command line — a bare pattern would SIGTERM the recipe shell and abort `stop`.
# The bracketed form is the same regex whose literal text no longer matches
# itself. ` [m]ain.py` (LEADING SPACE) hits the app driver `python main.py` and
# its `uv run` parent, but NOT ray's own `.../agent/main.py` (slash-prefixed).
# `-9` guarantees the driver dies (it survived plain SIGTERM in the wild); the
# explicit ray/gunicorn/PixInsight kills then reap the orphans a hard-killed
# main.py leaves behind (no atexit runs). `-` prefixes ignore "no match".
stop:
	-pkill -9 -f " [m]ain.py"
	-ray stop
	-pkill -f "[g]unicorn"
	-pkill -f "[P]ixInsight"
	@echo "✅ app stopped (main.py + ray + gunicorn + PixInsight)"

# Bounce the app: stop, let ray/sockets settle, start again (foreground).
restart: stop
	@sleep 2
	@$(MAKE) start
