# Technical Architecture & Implementation Guide

This document details the software architecture, dual-backend execution model, algorithms for Steps 1–3, optional extensions, and cross-platform build protocols.

---

## 1. Dual-Backend Architecture & Hardware Portability

To achieve maximum execution performance and hardware portability, the codebase follows a dual-backend design:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Python Orchestration & Policy Layer                  │
│       Agents, Epsilon Schedules, Sweeps, Logging, Plotting, Stats      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│     Backend A: Native C++ Engine  │       │   Backend B: PyTorch / CUDA (x64) │
│ (clang++ / MSVC / GCC + OpenMP)   │       │ Optional acceleration for NVIDIA  │
│  - Logit demand & capital dynamics│       │ RTX 4050 / x64 environments;      │
│  - Regulatory rules & audit flags │       │ tensor math & future DQN ablation │
│  - 10^6-state Value Iteration     │       │                                   │
│  - Primary engine for ARM64 & x64 │       │                                   │
└───────────────────────────────────┘       └───────────────────────────────────┘
```

### Hardware Support
- **ARM64 (Snapdragon X Elite / Copilot+ PC):** Runs Backend A via native `clang++` (`C:\msys64\clangarm64\bin\clang++.exe`) with `-O3 -fopenmp`. Bypasses PyTorch ARM64 wheel unavailability and leverages all CPU cores.
- **x64 (Intel / AMD + NVIDIA RTX 4050):** Runs Backend A natively, with Backend B optionally available via standard `pip install torch` for CUDA GPU acceleration.

---

## 2. Step 1: Exact Dynamic Programming Baseline (`src/benchmarks/exact_dp.py`)

### Reduced MDP Configuration
- $N = 3$ firms.
- $M \in \{6, 7, 8\}$ price levels.
- $T_{\text{ban}} \le 5$ periods.
- Capital grid: $\approx 20$ discrete levels in multiples of $\Phi$ over $[0, K_{\max}]$ ($K=0$ is absorbing).
- Total state space: $\approx 10^6$ states.

### Opponents & Bellman Operator
Opponents are fixed to honest reference strategies ($p^N$ or $p^M$). For firm 1:
$$V^*(s) = \max_{p_1 \in \mathcal{A}_1} \sum_{A \subseteq \mathcal{N}^{\text{act}}(s)} \mathbb{P}(\text{Aud} = A) \left[ R_1(p_1, \mathbf{p}_{-1}^{\text{hon}}, s, A) + \gamma V^*(s'(p_1, \mathbf{p}_{-1}^{\text{hon}}, s, A)) \right]$$

The C++ kernel executes this update using OpenMP thread parallelization across all available CPU cores until $\|V^{(k+1)} - V^{(k)}\|_\infty < \text{tol} \cdot \frac{1-\gamma}{2\gamma}$.

---

## 3. Step 2: Multi-Agent Tabular Q-Learning (`src/agents/q_learning.py`)

Each firm $i$ is an independent tabular Q-learner.

### State & Action Space
- Observation: $o_{i,t} = (\mathbf{p}_{t-1}, \mathbf{b}_t, \kappa_{i,t})$
  - Lagged prices $\mathbf{p}_{t-1} \in \mathcal{A}^3$ ($M = 11 \implies 1,331$ states).
  - Ban status $\mathbf{b}_t \in \{0, 1\}^3$ ($8$ states).
  - Capital bin $\kappa_{i,t} \in \{\text{low}, \text{mid}, \text{high}\}$:
    $$\kappa_{i,t} = \begin{cases} \text{low} & K_{i,t} \le 2\Phi \\ \text{mid} & 2\Phi < K_{i,t} \le 5\Phi \\ \text{high} & K_{i,t} > 5\Phi \end{cases}$$
- Total states per agent: $1,331 \times 8 \times 3 = 31,944 \approx 3.2 \times 10^4$.
- Action space: $M = 11$ price choices $\implies \approx 3.51 \times 10^5$ Q-values per agent (stored as contiguous `float32`).

### Exploration & Bellman Update
- Decaying $\varepsilon$-greedy: $\varepsilon_t = \max(\varepsilon_{\min}, \, \varepsilon_0 (1 - \beta_\varepsilon)^t)$.
- Standard update:
  $$Q_i(o_{i,t}, p_{i,t}) \leftarrow Q_i(o_{i,t}, p_{i,t}) + \alpha_t \left[ R_{i,t} + \gamma \max_{a'} Q_i(o_{i,t+1}, a') - Q_i(o_{i,t}, p_{i,t}) \right]$$

### Counterfactual Harm Engine (`src/agents/counterfactual.py`)
Harm is computed per period:
$$C_{\text{harm}}(i, t) = \sum_{j \ne i} \mathbb{I}\left(j \in \text{Aud}_t, \; F_j(\mathbf{p}_t) = 1, \; F_j(\mathbf{p}_t^{(i \to N)}) = 0\right)$$
Penalizes firm $i$ with $\mu_i$ only when substituting its Bertrand-Nash price $p_i^N$ clears rival $j$'s flag.

---

## 4. Step 3: Parameter Sweeps & Phase Diagrams

### Sweep Parameters
- Rules: Mean-Ratio, Median-Ratio, Price Cap, Z-Score.
- Ban lengths: $T_{\text{ban}} \in \{1, 3, 5, 10\}$.
- Initial capital ratios: $K_0 / \Phi \in [2.0, 20.0]$.
- Audit budgets: $B \in \{1, 2, 3\}$.
- Trait vectors: Sweeps over $\lambda \in [0, 0.5]$, $\mu \in [0, 1.0]$, and $\alpha \in [0, 2.0]$.
- Orchestration: Python `multiprocessing` executing $\ge 20$ seeds per configuration.

### Phase Diagrams & Learnability Gap
1. Plot 2D heatmaps of Weaponization Rate $W$ and Collusion Index $\Delta$ across $(K_0/\Phi, T_{\text{ban}})$.
2. Overlay the exact DP exploitability frontier ($\mathcal{E}_1 = 0$).
3. Identify the Learnability Gap: parameter regions where $\mathcal{E}_1 > 0$ under exact DP, but tabular Q-learners converge to $W \approx 0$.

---

## 5. Advanced Extensions

### Extension A: Slow-Timescale Adaptive Regulator (`src/extensions/learning_regulator.py`)
- Regulator operates on macro-epochs ($K_{\text{macro}} \gg 1$ periods) over history features $\mathcal{H}_t$ (price dispersion, ban frequency, price drift).
- Optimizes:
  $$\max_\phi \mathbb{E}\left[\sum_t \gamma_R^t (CS(\mathbf{p}_t) - \kappa |\text{Aud}_t| - \eta \text{FP}_t)\right]$$
- Checkpoints are frozen and tested against an adversarial best-response attacker.

### Extension B: Out-of-Distribution Demand Shock (`src/extensions/demand_shock.py`)
- For $t \ge T_{\text{shock}}$:
  $$Q_t = Q \cdot \max\{0, 1 + \psi \mathbb{I}(\bar{p}_t > \bar{p}_{t-1}) + \xi_t\}, \quad \xi_t \sim \mathcal{N}(0, \sigma_s^2)$$
- Evaluates welfare retention and bankruptcy rates on frozen trained policies under panic buying ($\psi > 0$).

---

## 6. Build, Testing & Verification Protocols

### Automated Build Script (`scripts/build_engine.py`)
The build script detects the environment and compiles the native engine:
```bash
python scripts/build_engine.py
```
- On ARM64 Windows: invokes `clang++ -O3 -shared -fopenmp src/csrc/engine.cpp -o src/csrc/engine.dll`.
- On x64 Windows/Linux: detects available compiler (`cl.exe`, `g++`, or `clang++`) and compiles accordingly.

### Test Requirements (`tests/`)
All PRs and modules must pass the automated test suite:
```bash
pytest tests/ -v
```
- `test_logit_engine.py`: Multinomial choice shares sum to $\le 1$ with outside option; LogSumExp stability.
- `test_capital_dynamics.py`: $K_{t+1} = \max\{0, K_t + \pi - \Phi\}$; absorbing barrier at $K = 0$.
- `test_proposition_1.py`: Exact algebraic validation of Proposition 1 conditions for Mean, Median, and Z-Score rules.
- `test_counterfactual_harm.py`: Zero penalty for honest/Nash pricing; causal verification of flags.

