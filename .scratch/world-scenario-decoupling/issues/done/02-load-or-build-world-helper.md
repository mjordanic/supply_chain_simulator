## Parent

`.scratch/world-scenario-decoupling/PRD.md`

## What to build

Add `load_or_build_world(name, build_fn, *, base_dir="data/worlds", force_rebuild=False, auto_confirm=False) -> World` to `world_builder.py` as the canonical script entry point for "load the cached world or build it with consent". Add a `LLMBuildAbortedError` exception class.

Behaviour:

- `path = Path(base_dir) / name / "world.json"`.
- If `path.exists()` and not `force_rebuild` → return `World.from_json(path)`. No prompt, no warning.
- Otherwise: print a two-line warning to `stderr` naming the missing path and noting the LLM cost (`5+ OpenAI calls`).
- If `auto_confirm=False`, call `input("  Press Enter to proceed, anything else to abort: ")`. Empty/whitespace response → proceed. Anything else → raise `LLMBuildAbortedError` with the user input quoted in the message. `KeyboardInterrupt` / `EOFError` (non-TTY contexts like CI / piped stdin) → raise `LLMBuildAbortedError` noting the exception type and that `auto_confirm=True` exists for non-interactive runs.
- Then call `build_fn()`, write via `world.to_json(path)`, return the world.

`force_rebuild` and `auto_confirm` are orthogonal: `force_rebuild=True` triggers the prompt; `auto_confirm=True, force_rebuild=False` proceeds silently on cache miss but still returns the cached file when present.

Use `print(..., file=sys.stderr)` for warning lines and `input(...)` for the prompt (renders in Jupyter cell output).

## Acceptance criteria

- [ ] `LLMBuildAbortedError` defined in `world_builder.py`.
- [ ] `load_or_build_world(name, build_fn, *, base_dir, force_rebuild, auto_confirm)` defined with the documented signature and defaults.
- [ ] Cache hit: file exists and `force_rebuild=False` → returns the saved world; `build_fn` is not called; no warning printed.
- [ ] Cache miss + `auto_confirm=True` → calls `build_fn()`, writes the file at `<base_dir>/<name>/world.json`, returns the new world.
- [ ] Cache miss + `auto_confirm=False` + simulated Enter (mocked `input` returning `""`) → calls `build_fn`, writes the file.
- [ ] Cache miss + `auto_confirm=False` + simulated abort (mocked `input` returning `"n"`) → raises `LLMBuildAbortedError`; `build_fn` is not called; no file is written.
- [ ] Cache miss + `auto_confirm=False` + non-TTY stdin (`EOFError`) → raises `LLMBuildAbortedError`; `build_fn` is not called; no file is written.
- [ ] `force_rebuild=True` + existing cache + `auto_confirm=True` → calls `build_fn` and overwrites the file.
- [ ] Warning is printed to `stderr` (not stdout) and names the missing path.
- [ ] Integration tests in `tests/llm/` exercise all six scenarios above with a monkeypatched `build_fn` returning a fixture world.

## Blocked by

- `01-world-serialisation-and-meta-block.md`
