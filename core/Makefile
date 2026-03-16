PYTEST = pytest
JUNIT_EXCLUSIVE = exclusive.xml
JUNIT_NON_EXCLUSIVE = non_exclusive.xml
JUNIT_MERGED = merged.xml

.PHONY: tests tests-full clean

tests:
	PYTHONPATH=. $(PYTEST) -n 4 --dist=loadscope -m "not exclusive and not slow" --junitxml=$(JUNIT_NON_EXCLUSIVE)
	PYTHONPATH=. $(PYTEST) -n 1 -m exclusive --junitxml=$(JUNIT_EXCLUSIVE)
	python merge_junitxml.py $(JUNIT_NON_EXCLUSIVE) $(JUNIT_EXCLUSIVE) > $(JUNIT_MERGED)
	@echo "✅ Tests complete. Merged report saved to $(JUNIT_MERGED)"

tests-full:
	PYTHONPATH=. $(PYTEST) -n 4 --dist=loadscope -m "not exclusive" --junitxml=$(JUNIT_NON_EXCLUSIVE)
	PYTHONPATH=. $(PYTEST) -n 1 -m exclusive --junitxml=$(JUNIT_EXCLUSIVE)
	python merge_junitxml.py $(JUNIT_NON_EXCLUSIVE) $(JUNIT_EXCLUSIVE) > $(JUNIT_MERGED)
	@echo "✅ Tests complete. Merged report saved to $(JUNIT_MERGED)"

clean:
	rm -f $(JUNIT_NON_EXCLUSIVE) $(JUNIT_EXCLUSIVE) $(JUNIT_MERGED)
