# Agent Rules — VisionServeX

**Future prompts can cite this file:** "Follow AGENT_RULES.md strictly."

These rules apply to Claude, Copilot, and any automated agent operating in the VisionServeX repository.

---

## Test commands

| Allowed | Forbidden |
|---------|-----------|
| `visionservex dev test quick` | `pytest` (bare, no marker filter) |
| `python scripts/test_quick_safe.py` | `pytest -n auto` |
| `python scripts/test_targeted_safe.py tests/test_foo.py` | Any pytest in background (`&`, `Popen`) |
| `python scripts/test_release_safe.py` (release only, once) | Running multiple pytest processes |

## Hard rules

1. **Never run pytest in the background.** `subprocess.Popen`, `&`, `nohup`, etc. are forbidden for pytest.
2. **Never run more than one test command at a time.** Wait for the previous run to finish.
3. **Never run the full test suite repeatedly.** Run quick tests iteratively; run full release validation once at the very end.
4. **Always check resources first.** Before any heavy operation run:
   ```
   visionservex dev resources
   ```
   or `python scripts/diagnose_resources.py`
5. **Stop immediately if:**
   - RAM usage > 80%
   - VRAM usage > 85% (free VRAM < 2 GB)
   - Disk free < 10 GB
   - Another pytest process is already running
6. **Never run real_model, gpu, or download tests** unless the user explicitly sets:
   ```
   VISIONSERVEX_RUN_REAL_MODEL_TESTS=1
   VISIONSERVEX_RUN_GPU_TESTS=1
   VISIONSERVEX_RUN_DOWNLOAD_TESTS=1
   ```
7. **Never use `pytest -n auto`** locally. Max workers = 1 for model/GPU/benchmark tests.
8. **Never use broad `find` or report generation** that writes huge outputs without a size limit.
9. **Test outputs go to tmp_path only.** Never write to `reports/`, `outputs/`, `indexes/`, or repo root during tests.
10. **Never silently download models** during a test run.

## Correct iterative development loop

```
# 1. Check resources
visionservex dev resources

# 2. Run quick tests (fast, < 60 s)
visionservex dev test quick

# 3. Run targeted test for what you changed
visionservex dev test targeted tests/test_foo.py

# 4. Repeat steps 2–3 until all quick tests pass

# 5. Only at the end: full release validation (once)
python scripts/diagnose_resources.py   # must pass
python scripts/test_release_safe.py
python -m build
python -m twine check dist/*
```

## Emergency recovery

If tests are consuming too much memory:

```bash
# Kill all VisionServeX tests (repo-scoped only)
visionservex dev kill-tests
# or
python scripts/kill_visionservex_tests.py

# Clean temp artifacts
visionservex dev clean-temp
visionservex dev clean-reports

# Check GPU VRAM
visionservex gpu status

# Force CUDA cache flush (Python REPL or script)
import torch; torch.cuda.empty_cache()
```

## See also

- `docs/agent_safety.md` — detailed safety rationale
- `src/visionservex/runtime/resource_guard.py` — implementation
- `scripts/diagnose_resources.py` — full diagnostic
