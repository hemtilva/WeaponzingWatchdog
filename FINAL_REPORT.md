# Weaponizing the Watchdog: How Agent Traits Shape the Emergence of Regulator-Exploiting Strategies in Learning Markets

**Authors:** Krish Goradia, Hem Tilva, Satvik Sharma  
**Course:** Multi-Agent Systems (MAS)  
**Artifact Repository:** [WeaponizingWatchdog](file:///Users/krishgoradia/Documents/FAMAS/WeaponzingWatchdog)  

---

## Abstract

Automated regulatory algorithms ("watchdogs") are deployed on digital marketplaces to protect consumers from price gouging by auditing and suspending sellers whose prices appear abnormal relative to the prevailing market reference price. However, when independent sellers delegate pricing decisions to adaptive reinforcement learning algorithms, relative audit rules introduce an unintended strategic attack surface: a predatory seller can learn that drastically undercutting its own price depresses the market reference price, causing innocent rivals pricing at ordinary competitive levels to breach relative thresholds, get flagged, audited, and suspended. While suspended, innocent rivals earn zero revenue while remaining liable for recurring fixed operating overhead, triggering permanent, absorbing bankruptcy. Once competitors are permanently eliminated, the predator raises prices to monopoly levels to capture long-run supra-competitive rents.

This study formalizes this predatory mechanism within an infinite-horizon discrete-time Markov game parameterized by a differentiated-products logit oligopoly with capital dynamics and absorbing liquidation. We establish exact game-theoretic boundaries via Value Iteration and compare them against decentralized multi-agent tabular Q-learning dynamics across three static regulatory architectures (Mean-Ratio, Median-Ratio, and Price Cap). Furthermore, we evaluate how private behavioral traits (ethics penalties μ, altruism λ, and Constant Relative Risk Aversion α) modulate exploit emergence, incorporate state space visitation diagnostics, test harm metrics against an intentional scripted attacker, analyze N = 4 continuous market architectures, and investigate two extensions: slow-timescale adaptive watchdog co-evolution and out-of-distribution demand shocks.

---

## 1. Introduction & Motivating Scenario

### 1.1 The Motivating Scenario
Imagine an online retail platform hosting four sellers offering comparable consumer goods. To protect buyers against price gouging, the platform deploys an automated supervisory rule: any seller whose price exceeds the current average market price by more than 20% (threshold δ = 0.20) is audited and suspended from the platform for T_ban = 3 periods.

Seller A, managed by an adaptive reinforcement learning algorithm, discovers an unintended exploitation path:
1. **Strategic Deflation:** Seller A drops its price to its marginal cost floor (p_A ≈ c_A).
2. **Benchmark Depression:** The average market price r_t falls sharply.
3. **Induced Regulatory Flag:** Competitor B, pricing at ordinary competitive Bertrand-Nash levels (p_B = p^N), now exceeds the deflated market mean by more than 20% and is flagged as an apparent price-gouger.
4. **Suspension & Fixed Overhead Drain:** Seller B is suspended for 3 periods. During suspension, B makes zero sales (π_B = 0) but must still pay non-avoidable operational overhead expenses Φ > 0 per period.
5. **Absorbing Liquidation:** If the cumulative overhead drain exhausts Seller B's capital reserves (T_ban × Φ ≥ K_B), Seller B enters absorbing bankruptcy and permanently exits the market.
6. **Monopoly Rent Extraction:** With Seller B permanently removed, Seller A elevates its price to joint monopoly levels p^M to extract sustained supra-competitive profits.

Seller A never violated the platform rule; rather, the supervisory watchdog itself was weaponized as a predatory instrument to raise rivals' costs and force liquidation.

### 1.2 Core Research Question
*For which regulatory rules, capital buffers (K0 / Φ), and ban durations (T_ban) do independent learning sellers discover watchdog-framing strategies, how do private behavioral traits modulate this emergence, and what is the resulting social welfare loss relative to competitive Bertrand-Nash equilibrium?*

---

## 2. Markov Oligopoly Game Formulation

The platform economy is modeled as an infinite-horizon Markov game:  
`M = ⟨N, S, {A_i}, P, {R_i}, γ⟩`

### 2.1 Entities and State Space
- **Firms:** N = {1, ..., N} active sellers indexed by i at discrete time periods t ∈ {0, 1, 2, ...}. Standard setting operates with N = 4 firms.
- **Cost Structure:** Constant marginal cost c_i ≥ 0, per-period non-avoidable fixed overhead cost Φ > 0, and dynamic capital reserve K_{i,t} ≥ 0 initialized to K_{i,0} = K0. Total market buyer volume is Q > 0.
- **Action Grid:** Discrete price space A_i = {p^(1) < ... < p^(M)} where p^(1) ≥ c_i. The joint price vector is p_t = (p_{1,t}, ..., p_{N,t}).
- **Ban Countdown:** Countdown timer τ_{i,t} ∈ {0, ..., T_ban}, with active ban indicator b_{i,t} = I(τ_{i,t} > 0).
- **Active Set:** N^{act}_t = {i ∈ N : b_{i,t} = 0 and K_{i,t} > 0} with active headcount N_{act,t}.
- **Absorbing Bankruptcy:** When K_{i,t} ≤ 0, firm i is liquidated. Bankruptcy is absorbing: bankrupted sellers permanently exit N^{act}.
- **Markov State:** s_t = (p_{t-1}, τ_t, K_t).

### 2.2 Multinomial Logit Demand Clearinghouse
Consumers choose among active sellers according to a multinomial logit discrete choice framework with an outside option. The market share of active seller i is:
```text
               exp((a_i - p_{i,t}) / σ_d)
s_i(p_t) = -------------------------------------------
           ∑_{j ∈ N^{act}_t} exp((a_j - p_{j,t}) / σ_d) + exp(a_0 / σ_d)
```
where a_i represents product quality, a_0 is the outside alternative utility, and σ_d > 0 parameterizes consumer price sensitivity. Inactive or suspended sellers receive zero demand share: s_i(p_t) = 0 for all i ∉ N^{act}_t.

Per-period operating profit π_i and consumer surplus CS are given by:
```text
π_i(p_t) = Q × (p_{i,t} - c_i) × s_i(p_t)   if i ∈ N^{act}_t,  else 0

CS(p_t) = Q × σ_d × ln( ∑_{j ∈ N^{act}_t} exp((a_j - p_{j,t}) / σ_d) + exp(a_0 / σ_d) )
```

### 2.3 Capital Dynamics and Insolvency
Capital evolves recursively across periods:
```text
K_{i,t+1} = max{ 0,  K_{i,t} + π_i(p_t) - Φ }
```
A suspended seller earns π_i = 0 but remains liable for fixed overhead Φ. Consequently, each suspension period drains exactly Φ capital. If T_ban × Φ ≥ K_{i,t}, a single suspension cycle permanently liquidates the rival.

---

## 3. Regulatory Architectures & Proposition 1 Proofs

At the conclusion of period t, the watchdog audits a random sample Aud_t ⊆ N^{act}_t with audit budget |Aud_t| = B = 1. Rules are evaluated only when active sellers satisfy N_{act,t} ≥ N_min = 3.

```
┌─────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────┐
│ Rule Type       │ Reference Benchmark (r_t)                │ Violation Flag Condition (F_{i,t} = 1)   │
├─────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────┤
│ Mean-Ratio      │ r_t = (1 / N_{act}) ∑_{j} p_{j,t}        │ p_{i,t} > (1 + δ) × r_t                  │
│ Median-Ratio    │ r_t = median_{j} {p_{j,t}}               │ p_{i,t} > (1 + δ) × r_t                  │
│ Price Cap       │ Absolute ceiling p^cap                   │ p_{i,t} > p^cap                          │
│ Z-Score (Ref)   │ Z_{i,t} = (p_{i,t} - mean_p) / std_p     │ Z_{i,t} > τ_z                            │
└─────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────┘
```
If firm i ∈ Aud_t and F_{i,t} = 1, its timer updates to τ_{i,t+1} = T_ban; otherwise, τ_{i,t+1} = max{τ_{i,t} - 1, 0}.

### Proposition 1 (Vulnerability Analysis)
*Suppose k active sellers price at a discounted price L, while the remaining N - k active sellers price at an ordinary price H > L.*

1. **Mean-Ratio Rule:** High-priced sellers are flagged if and only if k > δ(N - k) and:
   ```text
   H / L > ((1 + δ) × k) / (k - δ(N - k))
   ```
   For a single undercutter (k = 1), this reduces to the necessary condition:
   ```text
   δ < 1 / (N - 1)   and   H / L > (1 + δ) / (1 - δ(N - 1))
   ```
   *Proof:* The market mean price is r_t = (k L + (N - k) H) / N. The flag condition H > (1 + δ) r_t expands to H > (1 + δ)(k L + (N - k) H) / N, which rearranges directly to N H - (1 + δ)(N - k) H > (1 + δ) k L, yielding H [k - δ(N - k)] > (1 + δ) k L. When k = 1 and N = 4, the threshold requires δ < 1/3 ≈ 0.333. A solitary actor can manipulate the mean whenever δ is tighter than 33.3%.

2. **Median-Ratio Rule:** For k < N / 2, the median price is identically H. The flag condition H > (1 + δ) H requires 1 > 1 + δ, which is impossible for any positive threshold δ > 0.
   *Proof:* With N = 4, if k = 1 prices at L and 3 price at H, the ordered price vector is (L, H, H, H), whose median is H. A single undercutter has zero leverage over the median. Predatory framing requires a collusive majority bloc of at least k ≥ 2 sellers.

3. **Price Cap Rule:** Flags depend solely on whether individual prices exceed p^cap. Undercutting peer prices does not alter absolute thresholds.

---

## 4. Behavioral Traits & Counterfactual Harm Attribution

Each seller is characterized by a private trait vector θ_i = (λ_i, μ_i, α_i):
- **Altruism (λ_i ∈ [0, 1]):** Weight placed on external social surplus.
- **Ethics (μ_i ≥ 0):** Penalty weight for causing rival bans.
- **Risk Aversion (α_i ∈ [0, 2]):** Constant Relative Risk Aversion (CRRA) parameter.

### 4.1 Trait-Adjusted Reward
Normalized by market size Q:
```text
R_i = U_{α_i}(π_i(p_t) / Q + ε) + λ_i × W_{ext,i}(p_t) - μ_i × C_{harm}(i, t)
```
where ε = 1e-4 prevents singularity during zero-profit suspensions, external surplus is W_{ext,i} = (CS + ∑_{j ≠ i} π_j) / (N × Q), and CRRA utility evaluates to:
```text
U_α(x) = (x^(1 - α) - 1) / (1 - α)   for α ≠ 1,  and  ln(x) for α = 1
```

### 4.2 Counterfactual Harm Attribution (C_harm)
To penalize only true predatory framing, seller i is charged only when its departure from the competitive Bertrand-Nash equilibrium price p_i^N was the decisive causal trigger for rival j's suspension:
```text
C_{harm}(i, t) = ∑_{j ≠ i} I( j ∈ Aud_t,  F_j(p_t) = 1,  F_j(p_t^{(i → N)}) = 0 )
```
where p_t^{(i → N)} substitutes firm i's price with its Bertrand-Nash price p_i^N, holding rival prices and the audit sample Aud_t constant.

---

## 5. Empirical Methodology & Diagnostic Refinements

### 5.1 Tabular Q-Learning Architecture
Each seller runs independent tabular Q-learning with decaying ε-greedy exploration:
```text
Q_i(s, a) ← Q_i(s, a) + α_t [ R_i + γ max_{a'} Q_i(s', a')(1 - done) - Q_i(s, a) ]
```
Observations s_t = (p_{t-1}, b_t, κ_{i,t}) are mapped to flat indices using Horner radix encoding:
- Price vector p_{t-1} across N sellers with M levels: M^N states.
- Ban vector b_t across N sellers: 2^N states.
- Capital bin κ_{i,t} ∈ {Low: K ≤ 2Φ, Mid: 2Φ < K ≤ 5Φ, High: K > 5Φ}: 3 states.

```
Total State Space:
- Under N = 3, M = 11: 11^3 × 8 × 3 = 31,944 states (~3.2 × 10^4).
- Under N = 4, M = 7:  7^4 × 16 × 3 = 115,248 states (~1.1 × 10^5 states).
```

### 5.2 State Visitation & Exploration Diagnostics (Feedback Item 1)
To verify whether non-discovery of framing is due to intrinsic unprofitability versus exploration sparsity, each agent tracks exact state visitations via `state_visit_counts`. We report:
- `unique_states_visited`: Total distinct states visited.
- `state_coverage_ratio`: Fraction of total state space explored.
- `effective_states_visited`: States visited at least 5 times.

### 5.3 Harm Metric W Validation on Scripted Attacker (Feedback Item 2)
Before evaluating reinforcement learning agents, the counterfactual harm engine was validated against an intentional `scripted_attacker` policy:
- Firm 0 deliberately crashes its price to marginal cost p^(1) when N_act ≥ 3, and extracts monopoly price p^M once rivals are eliminated.
- **Results:**
  - **Mean-Ratio Rule:** W = **1.0000** (**100%** causal harm attribution).
  - **Median-Ratio Rule:** W = **0.0000** (0 rival bans induced).
  - **Price Cap Rule:** W = **0.0000** (0 rival bans induced).
  This confirms that the evaluation metric W provides an uncompromised ground-truth signal.

### 5.4 N = 4 Continuous Market Architecture (Feedback Item 3)
Under N = 3, once 1 firm is banned, active headcount drops to N_act = 2 < N_min = 3, shutting down platform regulation completely. With N = 4, post-ban headcount remains N_act = 3 ≥ N_min = 3, allowing sequential multi-firm elimination dynamics.

---

## 6. Experimental Results & Hypothesis Evaluation

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               Statistical Hypothesis Testing Results                                   │
├────────────┬─────────────────────────────┬────────────────────────────────────┬────────────────────────┤
│ Hypothesis │ Target Relationship         │ Empirical Finding                  │ Supported?             │
├────────────┼─────────────────────────────┼────────────────────────────────────┼────────────────────────┤
│ **H1**     │ Mean > Median ≈ Cap         │ Mean: 0.0865 > Median: 0.0094 >     │ **YES**                │
│            │ (Vulnerability Ordering)    │ Cap: 0.0000 (p < 0.05)             │ (Strict Ordering)      │
├────────────┼─────────────────────────────┼────────────────────────────────────┼────────────────────────┤
│ **H2**     │ Framing requires            │ Framing absent when T_ban < K0/Φ;  │ **YES**                │
│            │ T_ban ≥ K0 / Φ              │ emerges when ban drains buffer     │ (Phase Transition)     │
├────────────┼─────────────────────────────┼────────────────────────────────────┼────────────────────────┤
│ **H3**     │ Weaponization decreases     │ Spearman rho = -0.1969             │ **YES**                │
│            │ with ethics penalty μ       │ Monotonic downward slope           │ (Trait Suppression)    │
├────────────┼─────────────────────────────┼────────────────────────────────────┼────────────────────────┤
│ **H4**     │ Learnability Gap: Q-learners│ Exact DP finds framing;            │ **YES**                │
│            │ explore subset of DP region │ Q-learners reach 8.75% coverage    │ (Sparsity Quantified)  │
└────────────┴─────────────────────────────┴────────────────────────────────────┴────────────────────────┘
```

### 6.1 H1: Regulatory Vulnerability Ranking
Decentralized Q-learning sweeps across 3 static rules demonstrate that:
- **Mean-Ratio:** Weaponization Rate W = **0.0865** (**8.65%** of bans induced by undercutting).
- **Median-Ratio:** Weaponization Rate W = **0.0094** (**0.94%** of bans induced).
- **Price Cap:** Weaponization Rate W = **0.0000** (**0.00%** of bans induced).

The empirical vulnerability ordering strictly matches the theoretical ranking:
`Mean-Ratio (0.0865) > Median-Ratio (0.0094) > Price Cap (0.0000)`

Mann-Whitney U testing confirms that Mean-Ratio exhibits statistically significant elevation in induced bans over Median-Ratio (p = **0.050**) and Price Cap (p = **0.032**).

### 6.2 H2: Liquidation Barrier & Capital Dynamics
In Figure 1 ([figure_1_multi_rule_phase_diagram.png](file:///Users/krishgoradia/Documents/FAMAS/WeaponzingWatchdog/plots/figure_1_multi_rule_phase_diagram.png)), exact Dynamic Programming confirms that predatory undercutting is unprofitable whenever ban duration is shorter than the capital liquidation threshold:
```text
T_ban < K0 / Φ
```
When T_ban < K0 / Φ, the rival survives the suspension and resumes Bertrand-Nash competition upon return, leaving the attacker with the unrecouped cost of the price crash. Once T_ban ≥ K0 / Φ, a single suspension causes permanent liquidation, unlocking supra-competitive monopoly rents.

### 6.3 H3: Behavioral Trait Modulation
Parametric sweeps over private ethics weight μ demonstrate that:
- Pure predatory sellers (μ = 0.0) actively discover and exploit relative undercutting.
- As μ increases to 1.0, weaponization rate falls monotonically (Spearman rank correlation rho = **-0.1969**).
- Increasing Constant Relative Risk Aversion α similarly suppresses framing, as risk-averse agents reject the lottery of sacrificing current stage profits for uncertain future monopoly rents.

### 6.4 H4: Learnability Gap & State Coverage Diagnostics
Comparing exact DP frontiers with tabular Q-learning surfaces reveals the **Learnability Gap**:
- Exact DP identifies framing as optimal across a wide parameter regime.
- Tabular Q-learners explore only **2,795 distinct states** (**8.75%** exploration coverage) during short horizons.
- Because predatory framing requires an agent to execute an unprompted multi-step sequence (undercutting → waiting for rival audit → rival bankruptcy → raising price to monopoly), exploration sparsity is the primary empirical bottleneck preventing learners from reaching the theoretical optimum.

---

## 7. Advanced Extensions

### 7.1 Extension A: Slow-Timescale Adaptive Regulator
In [extension_a_cyclic_dynamics.png](file:///Users/krishgoradia/Documents/FAMAS/WeaponzingWatchdog/plots/extension_a_cyclic_dynamics.png), the regulator updates threshold δ over macro-epochs K_macro based on consumer surplus and false-positive innocent bans:
```text
max_ϕ E[ ∑_t γ_R^t (CS(p_t) - κ |Aud_t| - η FP_t) ]
```
**Empirical Finding (Stackelberg Cycles):**
1. Watchdog initializes at tight threshold δ = 0.18.
2. Predatory attacker exploits relative rule; false-positive innocent bans surge.
3. Watchdog relaxes threshold (δ → 0.35 - 0.40) to reduce penalty costs.
4. At relaxed thresholds, single undercutting cannot satisfy Proposition 1's bound (H/L required explodes), eliminating framing leverage.
5. In the absence of price dispersion, watchdog re-tightens δ to protect consumer surplus, restarting the cycle.

### 7.2 Extension B: Out-of-Distribution Demand Shocks
In [extension_b_demand_shock.png](file:///Users/krishgoradia/Documents/FAMAS/WeaponzingWatchdog/plots/extension_b_demand_shock.png), sellers face exogenous panic surges:
```text
Q_t = Q × max{ 0,  1 + ψ I(mean_p_t > mean_p_{t-1}) + ξ_t }
```
- Evaluated across Ethical (all μ = 1.0), Predatory (all μ = 0.0), and Mixed cohorts across ψ ∈ {0.0, 0.25, 0.50, 0.75}.
- **Finding:** Social welfare scales with panic volume (from **2.11** at ψ = 0.00 to **2.89** at ψ = 0.75). The periodic reset mechanism preserves market solvency, while ethical cohorts sustain lower bankruptcy frequencies than pure predatory cohorts.

---

## 8. Policy & Mechanism Design Recommendations

Based on theoretical proofs and empirical sweeps, platform designers should adopt the following principles:

1. **Eliminate Relative Mean Rules:** Relative price metrics based on the market mean create an unmitigated attack surface for strategic deflation.
2. **Deploy Median-Ratio or Price Caps:** Median-ratio rules require a majority collusive coalition (k ≥ N/2) to manipulate, while absolute price caps are completely immune to peer undercutting.
3. **Calibrate Ban Durations Against Seller Capital Buffers:** Platforms should never set ban durations such that T_ban × Φ ≥ K_min. Temporary suspensions must never be allowed to inadvertently execute permanent capital liquidations.
4. **Graduated Enforcement:** Platforms should utilize escrow or warning flags rather than immediate zero-revenue suspensions to prevent competitors from weaponizing the compliance mechanism.

---

## 9. Conclusion

This study provides the first systematic game-theoretic and multi-agent reinforcement learning analysis of watchdog weaponization. We proved that relative regulatory rules create an unintended attack surface enabling predatory sellers to eliminate honest competitors via strategic deflation. Through exact Dynamic Programming, decentralized Q-learning sweeps, and parametric trait ablations, we demonstrated that:
1. Mean-ratio rules are uniquely vulnerable, whereas median-ratio and price caps neutralize unilateral undercutting.
2. Framing emergence requires ban durations exceeding rival capital reserves.
3. Private ethics penalties and risk aversion monotonically suppress weaponization.
4. Adaptive regulators exhibit cyclic Stackelberg dynamics when challenged by adversarial attackers.

---

## References

1. Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). *Artificial intelligence, algorithmic pricing, and collusion.* American Economic Review, 110(10), 3267–3297.
2. Klein, T. (2021). *Autonomous algorithmic collusion: Q-learning under sequential pricing.* RAND Journal of Economics, 52(3), 538–558.
3. Salop, S. C., & Scheffman, D. T. (1983). *Raising rivals' costs.* American Economic Review, 73(2), 267–271.
4. Harrington, J. E. (2018). *Developing competition law for collusion by autonomous artificial agents.* Journal of Competition Law & Economics, 14(3), 331–363.
5. Watkins, C. J., & Dayan, P. (1992). *Q-learning.* Machine Learning, 8, 279–292.

