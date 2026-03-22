PYTEST = pytest
JUNIT_LIGHT = light.xml
JUNIT_HEAVY = heavy.xml
JUNIT_EXCLUSIVE = exclusive.xml
JUNIT_MERGED = merged.xml

.PHONY: tests tests-full clean

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
