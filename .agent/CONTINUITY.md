# Continuity Ledger

## [PLANS]

- 2026-05-18 [USER] Clean the codebase so it can replicate the paper results for: Voronoi cell visualization, cramming experiments, embedding estimation for support estimation, theorem bounds notebook, and copying task. Verify the code works, remove hard-coded local paths by hoisting configurable path globals to the beginning of scripts, and add run instructions.
- 2026-05-18 [ASSUMPTION] Treat existing dirty worktree changes as user-owned unless this task directly requires editing the same files.

## [DECISIONS]

- 2026-05-18 [ASSUMPTION] D001 ACTIVE: Use the existing experiment entrypoints where possible and keep changes focused on configurability, documentation, and smoke verification rather than redesigning the study.
- 2026-05-18 [CODE] D002 ACTIVE: Use repo-relative path globals and `LLM_VIS_MODEL_ROOT` instead of machine-specific absolute model/data roots.
- 2026-05-18 [CODE] D003 ACTIVE: Keep runner scripts as thin editable launchers; move parent-copy artifact/report helpers into `copying/parent_copy_artifacts.py` so `copying/run_copy_length_generalization.py` stays under 500 lines.

## [PROGRESS]

- 2026-05-18 [TOOL] No pre-existing `CONTINUITY.md` or `.agent/CONTINUITY.md` was found; this ledger was created for the workspace.
- 2026-05-18 [TOOL] Initial `git status --short` showed existing modified/deleted/untracked files, including `README.md`, `scripts/run_embedding_geometry.sh`, deleted `utils/*`, untracked `copying/`, `requirements.txt`, `scripts/run_copy_length_generalization.sh`, and `voronoi_visualizer/`.
- 2026-05-18 [CODE] Updated `README.md` with setup and run steps for Voronoi visualization, cramming, embedding sampling, theorem bounds, Voronoi/support notebook, and copying tasks.
- 2026-05-18 [CODE] Replaced hard-coded local roots in active scripts/modules with top-level globals or env-configurable values, including `download_llms.py`, `generate_embeddings.py`, `voronoi_visualizer/hf_llm.py`, cramming scripts, copying scripts, and notebooks.
- 2026-05-18 [CODE] Repaired stale runner targets: `scripts/run_embedding_geometry.sh` now calls `generate_embeddings.py`; `scripts/run_copy_length_generalization.sh` now calls `copying/run_copy_length_generalization.py`; cramming adaptive runner now loops over models instead of passing a comma-joined model id to HF.
- 2026-05-18 [CODE] Added `voronoi_visualizer/voronoi_volume.py` and updated notebook/package imports away from deleted `utils`/`src_clean` paths.
- 2026-05-18 [CODE] Cleared notebook outputs in `th_bounds_estimation.ipynb` and `cell_volume_tests.ipynb` so old local execution paths are not embedded in notebook output cells.
- 2026-05-18 [USER] User narrowed requested correction to only cleaning `scripts/run_copy_length_generalization.sh`.
- 2026-05-18 [CODE] Cleaned `scripts/run_copy_length_generalization.sh` into a single editable bash launcher with env-overridable variables and one call to `copying/run_copy_length_generalization.py`.
- 2026-05-18 [USER] User objected to README instructions referencing `copying/main.py`.
- 2026-05-18 [CODE] Removed the direct `python copying/main.py` copying-task block from `README.md`; copying instructions now point only to `scripts/run_copy_length_generalization.sh`.

## [DISCOVERIES]

- 2026-05-18 [TOOL] Path scan initially found machine-specific paths such as `/mnt/raid/mario/models/llms-theory`, `/mnt/mario/models/llms-theory`, and `/home/mario/codes/llm_vis/...`, plus stale `hidden_capacity`, `utils`, and `src_clean` references.
- 2026-05-18 [TOOL] Post-cleanup scan with `rg -n "(/home/|/Users/|/mnt/|/scratch/|/tmp/|~/|C:\\\\|\\.\\./|/workspace/|/local/|hidden_capacity|src_clean)"` over Python, shell, Markdown, notebooks, and requirements returned no matches.
- 2026-05-18 [TOOL] Verification passed: `python -m compileall` over edited Python files/packages; `bash -n` for the runner scripts; `--help` checks for main entrypoints; JSON validation for both notebooks; small smoke checks for Voronoi volume estimation and copying model-list parsing/forward pass.
- 2026-05-18 [CODE] `cell_convolution.py` now raises `FileNotFoundError` when required Voronoi CSVs are missing instead of silently writing placeholder rows.
- 2026-05-18 [TOOL] `bash -n scripts/run_copy_length_generalization.sh` passed; path scan on that script found no machine-specific absolute paths.

## [OUTCOMES]

- 2026-05-18 [TOOL] Major cleanup completed. Full paper-scale experiment execution remains UNCONFIRMED because it requires downloaded/gated HF models, GPU time, and generated data artifacts.
