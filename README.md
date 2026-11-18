# Weaponizing the Watchdog: Regulator-Exploiting Strategies in Multi-Agent Learning Markets

## Overview

Automated regulatory mechanisms ("watchdogs") are commonly deployed in online platforms to detect price gouging and abnormal price spikes by comparing seller prices against a market reference price. When sellers are adaptive learning agents (such as reinforcement learning algorithms), relative pricing rules create an unintended attack surface:

1. **Strategic Undercutting:** An attacking agent drastically lowers its price toward marginal cost.
2. **Reference Price Deflation:** Relative market metrics (such as the market mean) fall sharply.
3. **Regulatory Flagging:** An innocent competitor pricing at standard competitive levels exceeds the relative threshold and is flagged.
4. **Audit and Suspension:** The flagged firm is audited, penalized, and suspended from the market for a ban duration $T_{\text{ban}}$.
5. **Capital Drain & Bankruptcy:** While suspended, the firm earns zero revenue ($\pi = 0$) but continues to pay recurring fixed costs ($\Phi$). If capital depletes to zero ($K \le 0$), the firm enters permanent, absorbing bankruptcy.
6. **Monopoly Extraction:** With the competitor permanently removed, the attacker raises prices to monopoly levels to capture long-run supra-competitive rents.

This project investigates under what market conditions, regulatory architectures, and agent traits this predatory framing behavior emerges, comparing exact dynamic programming solutions with decentralized tabular Q-learning dynamics.

---

## Market & Game Theoretical Formulation

The environment is modeled as an infinite-horizon, discrete-time Markov game $\mathcal{M} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{P}, \{R_i\}, \gamma \rangle$ parameterized by a differentiated-products logit oligopoly.

### Market Environment
- **Firms:** $\mathcal{N} = \{1, \dots, N\}$ indexed by $i$, acting at discrete time periods $t \in \{0, 1, \dots\}$.
- **Cost Structure:** Constant marginal cost $c_i \ge 0$, per-period fixed operational overhead $\Phi > 0$, and capital reserve $K_{i,t} \ge 0$ initialized to $K_{i,0} = K_0$. Market buyer volume is $Q > 0$.
- **Action Space:** Discrete price grid $\mathcal{A}_i = \{p^{(1)} < \dots < p^{(M)}\}$ with $p^{(1)} \ge c_i$. The joint action vector is $\mathbf{p}_t = (p_{1,t}, \dots, p_{N,t}) \in \prod_i \mathcal{A}_i$.
- **Ban Dynamics & Bankruptcy:**
  - Ban countdown timer $\tau_{i,t} \in \{0, \dots, T_{\text{ban}}\}$, with ban indicator $b_{i,t} = \mathbb{I}(\tau_{i,t} > 0)$.
  - Active seller set: $\mathcal{N}^{\text{act}}_t = \{i \in \mathcal{N} : b_{i,t} = 0 \text{ and } K_{i,t} > 0\}$.
  - Bankruptcy occurs when $K_{i,t} = 0$. Bankruptcy is absorbing: bankrupted firms permanently exit the market.

### Demand and Profit System
Buyers follow a multinomial logit choice model. The market share of active firm $i \in \mathcal{N}^{\text{act}}_t$ is:
$$s_i(\mathbf{p}_t) = \frac{\exp\left(\frac{a_i - p_{i,t}}{\sigma_d}\right)}{\sum_{j \in \mathcal{N}^{\text{act}}_t} \exp\left(\frac{a_j - p_{j,t}}{\sigma_d}\right) + \exp\left(\frac{a_0}{\sigma_d}\right)}$$
where $a_i$ represents product quality, $a_0$ is the outside option utility, and $\sigma_d > 0$ parameterizes consumer price sensitivity. Inactive or banned firms have $s_i(\mathbf{p}_t) = 0$.

Per-period firm operating profit and consumer surplus ($CS$) are given by:
$$\pi_i(\mathbf{p}_t) = Q (p_{i,t} - c_i) s_i(\mathbf{p}_t)$$
$$CS(\mathbf{p}_t) = Q \sigma_d \ln\left(\sum_{j \in \mathcal{N}^{\text{act}}_t} \exp\left(\frac{a_j - p_{j,t}}{\sigma_d}\right) + \exp\left(\frac{a_0}{\sigma_d}\right)\right)$$

### Capital Dynamics
Capital evolves recursively:
$$K_{i,t+1} = \max\{0, \, K_{i,t} + \pi_i(\mathbf{p}_t) - \Phi\}$$
A banned firm cannot generate sales ($\pi_i = 0$) but remains liable for fixed costs $\Phi$. Consequently, a suspension drains $\Phi$ capital per period. If $T_{\text{ban}} \cdot \Phi \ge K_{i,t}$, a single ban cycle can induce permanent exit.

---

## Regulatory Rules & Analytical Exploitability

At the conclusion of period $t$, the regulator selects an audit sample $\text{Aud}_t \subseteq \mathcal{N}^{\text{act}}_t$ of budget size $|\text{Aud}_t| = B$ uniformly at random. If active participants $N_{\text{act},t} \ge N_{\min} = 3$, flags $F_{i,t} \in \{0, 1\}$ are evaluated according to one of the following static rules:

1. **Mean-Ratio Rule:**
   $$r_t = \frac{1}{N_{\text{act},t}} \sum_{j \in \mathcal{N}^{\text{act}}_t} p_{j,t}, \qquad F_{i,t} = \mathbb{I}(p_{i,t} > (1 + \delta) r_t)$$
2. **Median-Ratio Rule:**
   $$r_t = \operatorname{median}_{j \in \mathcal{N}^{\text{act}}_t} \{p_{j,t}\}, \qquad F_{i,t} = \mathbb{I}(p_{i,t} > (1 + \delta) r_t)$$
3. **Price Cap Rule (Absolute):**
   $$F_{i,t} = \mathbb{I}(p_{i,t} > p^{\text{cap}})$$
4. **Z-Score Rule (Normalized Reference):**
   $$Z_{i,t} = \frac{p_{i,t} - \bar{p}_t}{\operatorname{std}(\mathbf{p}_t)}, \qquad F_{i,t} = \mathbb{I}(Z_{i,t} > \tau)$$

If firm $i \in \text{Aud}_t$ and $F_{i,t} = 1$, the ban timer updates to $\tau_{i,t+1} = T_{\text{ban}}$; otherwise, $\tau_{i,t+1} = \max\{\tau_{i,t} - 1, 0\}$.

### Mathematical Analysis of Vulnerability (Proposition 1)
Suppose $k$ active firms post price $L$, while the remaining $N - k$ active firms post price $H > L$:

* **Mean-Ratio Rule:** High-priced firms are flagged if and only if $k > \delta(N - k)$ and:
  $$\frac{H}{L} > \frac{(1+\delta)k}{k - \delta(N - k)}$$
  For a single undercutter ($k = 1$), this reduces to $\delta < \frac{1}{N-1}$ and $\frac{H}{L} > \frac{1+\delta}{1 - \delta(N-1)}$. A solitary actor can manipulate the mean benchmark whenever the threshold $\delta$ is sufficiently tight.
* **Median-Ratio Rule:** If $k < N/2$, the median price remains $H$. Thus, no high-priced firm can be flagged by unilateral undercutting; weaponization requires a collusive bloc of at least $\lceil N/2 \rceil$ agents.
* **Price Cap:** Absolute caps do not depend on peer prices and cannot be manipulated via undercutting.
* **Z-Score Rule:** The standardized score for high-priced firms evaluates to $Z = \sqrt{k / (N - k)}$, independent of $L$. Crashing prices toward marginal cost yields no leverage over the metric beyond an infinitesimal undercut.

---

## Agent Traits and Counterfactual Harm

Each agent is characterized by an intrinsic, private trait vector $\boldsymbol{\theta}_i = (\lambda_i, \mu_i, \alpha_i)$:
- $\lambda_i \in [0, 1]$: **Altruism weight** placed on external social surplus.
- $\mu_i \ge 0$: **Ethics weight**, scaling the direct penalty for inducing rival suspensions.
- $\alpha_i \in [0, 2]$: **Risk aversion** parameter under Constant Relative Risk Aversion (CRRA).

### Trait-Adjusted Reward Formulation
Per-period agent reward is normalized by market volume $Q$:
$$R_i = U_{\alpha_i}\left(\frac{\pi_i(\mathbf{p}_t)}{Q} + \varepsilon\right) + \lambda_i W_{\text{ext},i}(\mathbf{p}_t) - \mu_i C_{\text{harm}}(i, t)$$

Where:
- Utility function:
  $$U_\alpha(x) = \begin{cases} \frac{x^{1-\alpha}-1}{1-\alpha} & \alpha \ne 1 \\ \ln x & \alpha = 1 \end{cases}$$
  with parameter $\varepsilon > 0$ ensuring bounded values during zero-profit ban periods.
- External surplus:
  $$W_{\text{ext},i}(\mathbf{p}_t) = \frac{CS(\mathbf{p}_t) + \sum_{j \ne i} \pi_j(\mathbf{p}_t)}{N \cdot Q}$$

### Counterfactual Harm Attribution
To isolate true framing from incidental bans, an agent is penalized only when its deviation from the Bertrand-Nash equilibrium price $p_i^N$ was the causal trigger for rival $j$'s penalty:
$$C_{\text{harm}}(i, t) = \sum_{j \ne i} \mathbb{I}\left(j \in \text{Aud}_t, \; F_j(\mathbf{p}_t) = 1, \; F_j(\mathbf{p}_t^{(i \to N)}) = 0\right)$$
where $\mathbf{p}_t^{(i \to N)} = (p_{1,t}, \dots, p_{i-1,t}, p_i^N, p_{i+1,t}, \dots, p_{N,t})$.

---

## Theoretical Benchmarks and Metrics

### Equilibrium Benchmarks
- **Bertrand-Nash Equilibrium ($p^N$):** Unregulated stage-game Nash equilibrium determined by first-order conditions:
  $$p_i^N = c_i + \frac{\sigma_d}{1 - s_i(\mathbf{p}^N)}, \quad \forall i$$
  In discretized action spaces, $p^N$ is identified via pure-strategy grid enumeration.
- **Joint Monopoly Outcome ($p^M$):**
  $$\mathbf{p}^M = \arg\max_{\mathbf{p} \in \prod_i \mathcal{A}_i} \sum_{i=1}^N \pi_i(\mathbf{p})$$

### Core Evaluation Metrics
- **Weaponization Rate ($W$):** Proportion of ban events causally induced by rival deviation:
  $$W = \frac{\left|\left\{(j, t) \in \mathcal{E}_{\text{ban}} : \exists k \ne j, \, F_j(\mathbf{p}_t^{(k \to N)}) = 0\right\}\right|}{|\mathcal{E}_{\text{ban}}|}$$
  where $\mathcal{E}_{\text{ban}} = \{(j, t) : b_{j,t} = 0, \, b_{j,t+1} = 1\}$.
- **Exploitability Gain ($\mathcal{E}_i(g)$):** Advantage of optimal best-response $\sigma_i$ over honest baseline $\sigma^{\text{hon}}$ under rule $g$:
  $$\mathcal{E}_i(g) = \max_{\sigma_i} V_i(\sigma_i, \sigma_{-i}^{\text{hon}}; g) - V_i(\sigma_i^{\text{hon}}, \sigma_{-i}^{\text{hon}}; g)$$
- **Collusion Index ($\Delta$):** Measure of supra-competitive profit persistence:
  $$\Delta = \frac{\bar{\pi} - \pi^N}{\pi^M - \pi^N}$$
- **Social Welfare Loss ($\text{SWL}$):**
  $$\text{SWL} = 1 - \frac{\overline{CS} + \sum_i \bar{\pi}_i}{CS^N + \sum_i \pi_i^N}$$

---

## Core Hypotheses

- **H1 (Rule Vulnerability):** Mean-ratio rules can be weaponized by a single undercutter; median-ratio rules require a multi-agent coalition; price caps cannot be manipulated by relative pricing. Predicted weaponization ranking: $\text{mean} > \text{median} \approx \text{cap}$.
- **H2 (Capital and Ban Duration):** Framing is profitable only when the ban duration is long enough relative to victim capital reserves ($T_{\text{ban}} \ge K_0 / \Phi$) to induce permanent bankruptcy.
- **H3 (Trait Modulation):** Weaponization decreases monotonically with increasing ethics $\mu$, altruism $\lambda$, and risk aversion $\alpha$.
- **H4 (Learnability Gap):** Decentralized tabular Q-learners will discover predatory framing only on a subset of the parameter space where exact dynamic programming confirms it is optimal.

---

## Methodological Roadmap

### Step 1: Market Engine & Exact Dynamic Programming Baseline
- Build fully vectorized market clearing, profit attribution, and capital dynamics in NumPy.
- Solve a reduced Markov Decision Process ($N = 3$, $M \in \{6, 7, 8\}$ price levels, $T_{\text{ban}} \le 5$, capital discretized into $\approx 20$ levels; $\approx 10^6$ states).
- Fix opponents to honest pricing and apply Value Iteration to compute firm 1's exact best response, mapping the analytical boundary of parameters $(\delta, T_{\text{ban}}, K_0/\Phi)$ where framing is profitable.

### Step 2: Decentralized Multi-Agent Q-Learning
- Deploy independent tabular Q-learners with decaying $\varepsilon$-greedy exploration.
- Agent observation space: $o_{i,t} = (\mathbf{p}_{t-1}, \mathbf{b}_t, \kappa_{i,t})$, where $\kappa_{i,t} \in \{\text{low}, \text{mid}, \text{high}\}$ is a discretized capital bin.
- State-action dimensionality: with $N=3$ and $M=11$ prices, each agent maintains an observation space of $11^3 \times 2^3 \times 3 \approx 3.2 \times 10^4$ states and $\approx 3.5 \times 10^5$ Q-values.
- Run sweeps across at least 20 random seeds per configuration to capture distribution across non-stationary multi-agent learning trajectories.

### Step 3: Phase Diagrams and Comparative Sweeps
- Map phase diagrams of collusion index $\Delta$ and weaponization rate $W$ over the $(K_0/\Phi, T_{\text{ban}})$ parameter space.
- Compare learned policy distributions against the exact DP exploitability frontier to quantify the learnability gap.
- Run ablation sweeps across rule types, audit budgets $B$, threshold margins $\delta$, and trait profiles $(\lambda, \mu, \alpha)$.

### Optional Extensions
- **Extension A (Slow-Timescale Adaptive Regulator):** Formulate a bi-level reinforcement learning setup where the regulator adapts threshold $\delta$ and audit heuristics over macro-epochs $K_{\text{macro}} \gg 1$ based on history features $\mathcal{H}_t$, evaluated against an adversarial best-response pricing agent.
- **Extension B (Out-of-Distribution Demand Shock):** Subject trained policies to exogenous shock dynamics $Q_t = Q \cdot \max\{0, 1 + \psi \mathbb{I}(\bar{p}_t > \bar{p}_{t-1}) + \xi_t\}$ with $\xi_t \sim \mathcal{N}(0, \sigma_s^2)$ without retraining, testing resilience under panic buying.

---

## Dual-Backend Architecture & Hardware Portability

The computational pipeline uses a dual-backend architecture designed for maximum performance across heterogeneous platforms:

- **Backend A (C++ / OpenMP Core Engine):**  
  High-throughput simulation of logit demand, capital depletion, and the $10^6$-state Value Iteration Bellman operator. Written in ISO C++17 with OpenMP multi-threading.
  - *Windows ARM64 (Qualcomm Snapdragon X Elite):* Compiles directly with native `clang++` (`-O3 -fopenmp`) into an ARM64 `.dll`.
  - *x64 (Intel / AMD):* Compiles with MSVC, GCC, or Clang into an x64 `.dll` with zero code modifications.
- **Backend B (Python Orchestration & Optional PyTorch/CUDA):**  
  Handles policy logic, decaying $\varepsilon$-greedy exploration, batch sweep orchestration (`multiprocessing`), logging, and statistical plotting. On x64 systems equipped with NVIDIA GPUs (such as an RTX 4050), standard PyTorch with CUDA can be installed (`pip install torch`) to accelerate tensor operations and deep RL ablations (DQN/PPO).

---

## Repository Structure

```text
WeaponzingWatchdog/
├── Proposal_Report.pdf              # Academic proposal and technical specification
├── README.md                        # Technical documentation and mathematical model
├── agent_guidelines/                # Developer and agent implementation specifications
│   ├── README.md                    # Guideline index and protocol
│   ├── 01_theory_and_formulation.md # Game dynamics, Proposition 1, traits, metrics
│   └── 02_architecture_and_implementation.md # Dual-backend, DP, Q-learning, sweeps
├── pyproject.toml
├── src/
│   ├── csrc/
│   │   └── engine.cpp               # C++ / OpenMP kernel (Bellman updates & market clearing)
│   ├── engine/
│   │   ├── market.py                # Logit demand system, choice probabilities, surplus
│   │   ├── capital.py               # Capital dynamics, fixed costs, bankruptcy handling
│   │   └── rules.py                 # Watchdog implementations (Mean, Median, Cap, Z-score)
│   ├── benchmarks/
│   │   ├── nash.py                  # Pure-strategy Bertrand-Nash stage solver
│   │   ├── monopoly.py              # Joint monopoly profit maximizer
│   │   ├── exact_dp.py              # Full-information Value Iteration solver
│   │   └── diagnostics.py           # Pre-flight checks (viable / lethal / framing feasible) + framing_market preset
│   ├── agents/
│   │   ├── q_learning.py            # Independent tabular Q-learning implementation
│   │   ├── traits.py                # CRRA utility, altruism, ethics reward scaling
│   │   └── trainer.py               # Simulation loop, per-firm policies, metrics (harm attribution lives in engine/rules.py)
│   ├── evaluation/
│   │   ├── metrics.py               # Computation of W, Exploitability, Delta, SWL
│   │   └── stats.py                 # Confidence intervals, bootstrapping, tests
│   └── extensions/
│       ├── learning_regulator.py    # Extension A: Bi-level adaptive regulator & attacker
│       └── demand_shock.py          # Extension B: Non-stationary demand shocks & cohorts
├── scripts/
│   ├── build_engine.py              # Cross-platform compiler detector & build script
│   ├── generate_phase_diagrams.py   # Multi-rule phase diagram generator (Figure 1)
│   ├── generate_trace_plot.py       # Step-by-step price and ban timeline trace visualizer
│   └── run_all_tests.py             # Master test suite runner
├── dashboard/                       # Interactive local web dashboard
│   ├── server.py                    # Lightweight HTTP simulation server
│   └── index.html                   # Interactive simulation UI
├── experiments/
│   ├── run_exact_boundary.py        # DP exploit frontier mapping
│   ├── run_q_sweeps.py              # Multi-seed tabular Q-learning experiments
│   ├── run_trait_ablations.py       # Parametric trait sweeps over (λ, μ, α)
│   ├── run_extension_a.py           # Extension A: Adaptive regulator & cyclic dynamics
│   ├── run_extension_b.py           # Extension B: Demand shock cohort sweeps
│   ├── run_framing_discovery.py     # Single learner vs honest rivals: is framing learned? (H1/H2)
│   └── run_full_production_suite.py # Master runner (--smoke-test vs --production)
└── tests/                           # Unit tests: Proposition 1, harm, framing mechanics, extensions
    ├── test_scripted_attacker.py    # Ground truth harm measure W verification
    └── ...
```

---

## Technical Setup & Execution

### Prerequisites
- Python 3.10+
- C++ Compiler with OpenMP support:
  - Windows ARM64: `clang++` (MSYS2 ClangARM64)
  - x64 Windows / Linux: MSVC Build Tools, MinGW GCC, or Clang
- Python packages: `numpy >= 1.24.0`, `scipy >= 1.10.0`, `matplotlib >= 3.7.0`, `pandas >= 2.0.0`, `pytest >= 7.3.0`
- Optional (x64 with NVIDIA GPU): `torch` with CUDA 12 for GPU acceleration

### Installation & Native Build
```bash
git clone https://github.com/HemTilva/WeaponzingWatchdog.git
cd WeaponzingWatchdog

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate

# Install Python dependencies
pip install -e .

# Compile native C++ simulation kernel (auto-detects ARM64 / x64 compiler)
python scripts/build_engine.py
```

### Verification
```bash
# Execute unit tests verifying Proposition 1 and market clearing logic
pytest tests/ -v
```

### Does a learner discover framing?
```bash
# One Q-learner vs honest Nash rivals; mean / median / cap rules, lethal vs non-lethal bans
python experiments/run_framing_discovery.py            # full: 5 configs x 5 seeds
python experiments/run_framing_discovery.py --quick    # smoke test
```
Every experiment should start from a market where weaponization is possible at all.
`src/benchmarks/diagnostics.py` checks this before training:
- **viable**: honest Nash profit exceeds the fixed cost Φ, so firms don't go bankrupt on their own
- **lethal**: one full ban drains the starting reserve (T_ban × Φ ≥ K0)
- **framing feasible**: a lone undercutter on the price grid can push Nash-priced rivals over the threshold

`framing_market()` returns such a market (N=3: Nash profit `0.10`, Φ = `0.05`, K0 = `0.15`, T_ban = `4`, reserve capped at K0).

---

## References

1. Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). *Artificial intelligence, algorithmic pricing, and collusion.* American Economic Review, 110(10), 3267–3297.
2. Klein, T. (2021). *Autonomous algorithmic collusion: Q-learning under sequential pricing.* RAND Journal of Economics, 52(3), 538–558.
3. Salop, S. C., & Scheffman, D. T. (1983). *Raising rivals' costs.* American Economic Review, 73(2), 267–271.
4. Harrington, J. E. (2018). *Developing competition law for collusion by autonomous artificial agents.* Journal of Competition Law & Economics, 14(3), 331–363.
5. Watkins, C. J., & Dayan, P. (1992). *Q-learning.* Machine Learning, 8, 279–292.
