# Archive: superseded code (not maintained, not used as evidence)

This folder keeps early project code for the record. **Nothing here produces a result in
the report.** Every table and figure in `paper/` comes from the scripts listed in the main
README's reproduction table, which live in `experiments/`, `scripts/` and `src/`.

What is here and why it was retired:

| Path | What it was | Why it is archived |
|---|---|---|
| `src/extensions/`, `experiments/run_extension_a.py`, `run_extension_b.py` | Extension A (learning regulator) and B (demand shock) | Run only before the fix to the learning dynamics; never rerun. |
| `src/agents/dqn_agent.py`, `dqn_trainer.py`, `experiments/run_dqn_ablation.py` | First DQN ablation | Replaced by `src/agents/dqn_single.py` and `dqn_multi.py`. |
| `src/evaluation/tournament.py`, `experiments/run_tournament.py` | Evolutionary tournament | Matches were too short for agents to learn; never rerun. |
| `experiments/run_q_sweeps.py`, `run_trait_ablations.py`, `run_full_production_suite.py`, `run_exact_boundary.py`, `run_defended_watchdog.py`, `run_framing_discovery.py`, `run_all_learners.py` | Early sweep drivers | Replaced by `experiments/run_study.py` and the dedicated study scripts. |
| `scripts/generate_phase_diagrams.py`, `generate_trace_plot.py`, `run_all_tests.py`, `launch_production.sh` | Early plotting and launch helpers | Replaced by `scripts/make_paper_figures.py` and plain `pytest`. |
| `dashboard/` | Local interactive viewer | Not used by the study. |
| `plots/` | Figures from earlier runs (`plots/legacy/` predates the learning-dynamics fix) | Superseded by `paper/figures/`. |
| `tests/` | Tests of the code above | Not collected by `pytest` (the project's `testpaths` is `tests/`). |

These files were moved here unchanged. Scripts locate the repository relative to their own
folder, so some will not run from this location without editing.
