# Agent Guidelines

This directory contains the core technical guidelines for agents developing the "Weaponizing the Watchdog" research codebase.

## Documentation Structure

1. **[01_theory_and_formulation.md](file:///d:/hem/project/WeaponzingWatchdog/agent_guidelines/01_theory_and_formulation.md)**  
   Theoretical foundations, Markov game formalization, logit demand clearing, capital dynamics and bankruptcy, static regulatory rules, Proposition 1 algebraic proofs, trait-parameterized reward objectives, counterfactual harm attribution, and core hypotheses.

2. **[02_architecture_and_implementation.md](file:///d:/hem/project/WeaponzingWatchdog/agent_guidelines/02_architecture_and_implementation.md)**  
   Dual-backend execution model (C++ OpenMP engine + Python / PyTorch CUDA interface), Step 1 dynamic programming (Value Iteration) solver, Step 2 multi-agent tabular Q-learning, Step 3 parameter sweeps and phase diagrams, advanced extensions (A & B), and cross-platform build/testing standards.

## Execution Rules for Agents

- Review `01_theory_and_formulation.md` before altering mathematical or behavioral logic.
- Review `02_architecture_and_implementation.md` before adding or modifying engine kernels, algorithms, or sweep pipelines.
- Ensure all inner simulation loops remain vectorized in NumPy or implemented in the C++ OpenMP backend.
- Maintain full cross-platform compatibility across Windows ARM64 (Snapdragon) and x64 (NVIDIA RTX / CUDA).

