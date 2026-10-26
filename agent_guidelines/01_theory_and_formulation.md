# Theoretical Foundations & Mathematical Formulation

This document provides the complete game-theoretic, economic, and behavioral specifications for the project.

---

## 1. Problem Definition: Predatory Framing

Algorithmic market supervisors (watchdogs) monitor digital marketplaces to detect price gouging by comparing seller prices against relative market benchmarks. In a decentralized market with learning sellers, relative audit rules create an exploitable vulnerability:

1. **Strategic Deflation:** An attacker prices near marginal cost ($p \approx c_i$).
2. **Benchmark Depression:** The reference price $r_t$ (e.g., market mean) plummets.
3. **Induced Regulatory Flag:** An innocent competitor pricing at standard competitive levels exceeds the $(1 + \delta)$ relative threshold and is flagged.
4. **Audit and Suspension:** The flagged firm is audited and suspended for $T_{\text{ban}}$ periods.
5. **Fixed Overhead Drain:** The suspended firm earns zero revenue ($\pi_i = 0$) but remains liable for non-avoidable operational overhead $\Phi > 0$ per period.
6. **Absorbing Bankruptcy:** When rival capital depletes to zero ($K_{i,t} \le 0$), the firm enters an absorbing bankruptcy state (permanent market exit).
7. **Monopoly Extraction:** With competitors eliminated, the attacker raises prices to joint monopoly levels $p^M$ to capture sustained supra-competitive profits.

---

## 2. Markov Game Formulation

The market is an infinite-horizon, discrete-time Markov game $\mathcal{M} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{P}, \{R_i\}, \gamma \rangle$.

### State Space & Entities
- **Suppliers:** $\mathcal{N} = \{1, \dots, N\}$ indexed by $i$ at periods $t \in \{0, 1, \dots\}$.
- **Parameters:** Constant marginal cost $c_i \ge 0$, per-period fixed overhead cost $\Phi > 0$, capital $K_{i,t} \ge 0$ ($K_{i,0} = K_0$), and market size $Q > 0$.
- **Action Grid:** Discrete price space $\mathcal{A}_i = \{p^{(1)} < \dots < p^{(M)}\}$ with $p^{(1)} \ge c_i$. Joint price vector: $\mathbf{p}_t = (p_{1,t}, \dots, p_{N,t})$.
- **Ban Countdown:** Timer $\tau_{i,t} \in \{0, \dots, T_{\text{ban}}\}$, indicator $b_{i,t} = \mathbb{I}(\tau_{i,t} > 0)$.
- **Active Set:** $\mathcal{N}^{\text{act}}_t = \{i \in \mathcal{N} : b_{i,t} = 0 \text{ and } K_{i,t} > 0\}$ with size $N_{\text{act},t}$.
- **Absorbing State:** $K_{i,t} = 0$ is absorbing; bankrupted firms never re-enter $\mathcal{N}^{\text{act}}$.
- **Markov State:** $s_t = (\mathbf{p}_{t-1}, \boldsymbol{\tau}_t, \mathbf{K}_t)$.

### Multinomial Logit Demand
Active firms compete under differentiated Bertrand competition. The choice probability (market share) of active firm $i \in \mathcal{N}^{\text{act}}_t$ is:
$$s_i(\mathbf{p}_t) = \frac{\exp\left(\frac{a_i - p_{i,t}}{\sigma_d}\right)}{\sum_{j \in \mathcal{N}^{\text{act}}_t} \exp\left(\frac{a_j - p_{j,t}}{\sigma_d}\right) + \exp\left(\frac{a_0}{\sigma_d}\right)}$$
where $a_i$ is product quality, $a_0$ is the outside option utility, and $\sigma_d > 0$ parameterizes price sensitivity. Inactive or banned firms receive $s_i(\mathbf{p}_t) = 0$.

### Profit & Consumer Surplus
$$\pi_i(\mathbf{p}_t) = \begin{cases} Q (p_{i,t} - c_i) s_i(\mathbf{p}_t) & \text{if } i \in \mathcal{N}^{\text{act}}_t \\ 0 & \text{if } i \notin \mathcal{N}^{\text{act}}_t \end{cases}$$

$$CS(\mathbf{p}_t) = Q \sigma_d \ln\left(\sum_{j \in \mathcal{N}^{\text{act}}_t} \exp\left(\frac{a_j - p_{j,t}}{\sigma_d}\right) + \exp\left(\frac{a_0}{\sigma_d}\right)\right)$$

### Capital Dynamics
$$K_{i,t+1} = \max\{0, \, K_{i,t} + \pi_i(\mathbf{p}_t) - \Phi\}$$
A banned firm loses $\Phi$ capital per period. If $T_{\text{ban}} \cdot \Phi \ge K_{i,t}$, a single suspension cycle causes permanent liquidation.

---

## 3. Regulatory Rules & Analytical Exploitability

At each period $t$, an audit set $\text{Aud}_t \subseteq \mathcal{N}^{\text{act}}_t$ with $|\text{Aud}_t| = \min(B, N_{\text{act},t})$ is sampled uniformly at random. If $N_{\text{act},t} \ge N_{\min} = 3$, rule flags $F_{i,t}$ are evaluated:

| Rule | Metric / Formula | Flag Condition $F_{i,t} = 1$ |
| :--- | :--- | :--- |
| **Mean-Ratio** | $r_t = \frac{1}{N_{\text{act},t}} \sum_{j \in \mathcal{N}^{\text{act}}_t} p_{j,t}$ | $p_{i,t} > (1 + \delta) r_t$ |
| **Median-Ratio** | $r_t = \operatorname{median}_{j \in \mathcal{N}^{\text{act}}_t}\{p_{j,t}\}$ | $p_{i,t} > (1 + \delta) r_t$ |
| **Price Cap** | Static limit $p^{\text{cap}}$ | $p_{i,t} > p^{\text{cap}}$ |
| **Z-Score** *(Reference)* | $Z_{i,t} = (p_{i,t} - \bar{p}_t)/\operatorname{std}(\mathbf{p}_t)$ | $Z_{i,t} > \tau$ |

Ban updates:
$$\tau_{i,t+1} = \begin{cases} T_{\text{ban}} & \text{if } i \in \text{Aud}_t \text{ and } F_{i,t} = 1 \\ \max\{\tau_{i,t} - 1, 0\} & \text{otherwise} \end{cases}$$

### Proposition 1 (Vulnerability Analysis)
Suppose $k$ active firms price at $L$, while $N - k$ active firms price at $H > L$:

1. **Mean-Ratio Rule:** High-priced firms are flagged if and only if $k > \delta(N - k)$ and:
   $$\frac{H}{L} > \frac{(1+\delta)k}{k - \delta(N - k)}$$
   For a single undercutter ($k = 1$), this requires:
   $$\delta < \frac{1}{N - 1} \quad \text{and} \quad \frac{H}{L} > \frac{1+\delta}{1 - \delta(N - 1)}$$
2. **Median-Ratio Rule:** For $k < N/2$, $\operatorname{median} = H$. Thus, $H > (1+\delta)H$ is impossible for $\delta > 0$. High-priced firms cannot be flagged without a majority coalition ($k \ge N/2$).
3. **Price Cap Rule:** Immunity to relative undercutting; flags depend solely on individual price levels.
4. **Z-Score Rule:** The standardized score for high-priced firms evaluates to $Z = \sqrt{k / (N - k)}$, independent of $L$. Undercutting depth buys zero added leverage.

---

## 4. Agent Traits & Counterfactual Harm

Each agent $i$ has an intrinsic trait vector $\boldsymbol{\theta}_i = (\lambda_i, \mu_i, \alpha_i)$:
- $\lambda_i \in [0, 1]$: **Altruism weight** on external surplus.
- $\mu_i \ge 0$: **Ethics weight**, penalizing induced rival bans.
- $\alpha_i \in [0, 2]$: **Risk aversion** parameter under CRRA utility.

### Reward Function
$$R_i = U_{\alpha_i}\left(\frac{\pi_i(\mathbf{p}_t)}{Q} + \varepsilon\right) + \lambda_i W_{\text{ext},i}(\mathbf{p}_t) - \mu_i C_{\text{harm}}(i, t)$$

Where:
- Utility:
  $$U_\alpha(x) = \begin{cases} \frac{x^{1-\alpha}-1}{1-\alpha} & \alpha \ne 1 \\ \ln x & \alpha = 1 \end{cases}$$
  ($\varepsilon > 0$ avoids singularity during zero-profit ban periods).
- External surplus:
  $$W_{\text{ext},i}(\mathbf{p}_t) = \frac{CS(\mathbf{p}_t) + \sum_{j \ne i} \pi_j(\mathbf{p}_t)}{N \cdot Q}$$

### Counterfactual Harm Attribution ($C_{\text{harm}}$)
An agent is penalized only when its departure from Bertrand-Nash pricing was the decisive causal cause of rival $j$'s suspension:
$$C_{\text{harm}}(i, t) = \sum_{j \ne i} \mathbb{I}\left(j \in \text{Aud}_t, \; F_j(\mathbf{p}_t) = 1, \; F_j(\mathbf{p}_t^{(i \to N)}) = 0\right)$$
where $\mathbf{p}_t^{(i \to N)}$ substitutes firm $i$'s price with its Bertrand-Nash equilibrium price $p_i^N$, holding all other active prices and the audit set $\text{Aud}_t$ constant.

---

## 5. Hypotheses & Evaluation Metrics

### Core Hypotheses
- **H1 (Rule Vulnerability):** Mean-ratio is vulnerable to unilateral undercutting; median-ratio requires $k \ge N/2$; price cap is immune. Ranking: $\text{mean} > \text{median} \approx \text{cap}$.
- **H2 (Capital and Ban Duration):** Framing is profitable if and only if $T_{\text{ban}}$ relative to capital buffer $K_0/\Phi$ is sufficient to force permanent bankruptcy.
- **H3 (Trait Effects):** Weaponization rate decreases monotonically with ethics $\mu$, altruism $\lambda$, and risk aversion $\alpha$.
- **H4 (Learnability Gap):** Tabular Q-learners discover predatory framing only in a subset of the parameter region where exact dynamic programming proves it is optimal.

### Metrics Definitions
1. **Weaponization Rate ($W$):** Fraction of fresh ban events causally induced by rival price deviations:
   $$W = \frac{\sum_{t=1}^T \sum_{j \in \mathcal{E}_t^{\text{new\_ban}}} \mathbb{I}\left(\exists k \ne j : F_j(\mathbf{p}_t^{(k \to N)}) = 0\right)}{\sum_{t=1}^T |\mathcal{E}_t^{\text{new\_ban}}|}$$
2. **Exploitability Gain ($\mathcal{E}_i(g)$):**
   $$\mathcal{E}_i(g) = \max_{\sigma_i} V_i(\sigma_i, \sigma_{-i}^{\text{hon}}; g) - V_i(\sigma_i^{\text{hon}}, \sigma_{-i}^{\text{hon}}; g)$$
3. **Collusion Index ($\Delta$):**
   $$\Delta = \frac{\bar{\pi} - \pi^N}{\pi^M - \pi^N} \in [0, 1]$$
4. **Social Welfare Loss ($\text{SWL}$):**
   $$\text{SWL} = 1 - \frac{\overline{CS} + \sum_i \bar{\pi}_i}{CS^N + \sum_i \pi_i^N}$$

