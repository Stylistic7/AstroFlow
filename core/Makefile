PYTEST = pytest
JUNIT_LIGHT = light.xml
JUNIT_HEAVY = heavy.xml
JUNIT_EXCLUSIVE = exclusive.xml
JUNIT_MERGED = merged.xml

.PHONY: tests tests-full clean ray-clean ray-status

tests:
	PYTHONPATH=. $(PYTEST) -n auto --dist=loadscope -m "not exclusive and not slow and not heavy" --junitxml=$(JUNIT_LIGHT)
	PYTHONPATH=. $(PYTEST) -n 1 -m "heavy and not exclusive and not slow" --junitxml=$(JUNIT_HEAVY)
	PYTHONPATH=. $(PYTEST) -n 1 -m exclusive --junitxml=$(JUNIT_EXCLUSIVE)
	python merge_junitxml.py $(JUNIT_LIGHT) $(JUNIT_HEAVY) $(JUNIT_EXCLUSIVE) > $(JUNIT_MERGED)
	@echo "✅ Tests complete. Merged report saved to $(JUNIT_MERGED)"

tests-full:
	PYTHONPATH=. $(PYTEST) -n auto --dist=loadscope -m "not exclusive and not heavy" --junitxml=$(JUNIT_LIGHT)
	PYTHONPATH=. $(PYTEST) -n 1 -m "heavy and not exclusive" --junitxml=$(JUNIT_HEAVY)
	PYTHONPATH=. $(PYTEST) -n 1 -m exclusive --junitxml=$(JUNIT_EXCLUSIVE)
	python merge_junitxml.py $(JUNIT_LIGHT) $(JUNIT_HEAVY) $(JUNIT_EXCLUSIVE) > $(JUNIT_MERGED)
	@echo "✅ Tests complete. Merged report saved to $(JUNIT_MERGED)"

clean:
	rm -f $(JUNIT_LIGHT) $(JUNIT_HEAVY) $(JUNIT_EXCLUSIVE) $(JUNIT_MERGED)

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
