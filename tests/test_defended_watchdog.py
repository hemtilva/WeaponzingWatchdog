"""Unit tests for Defended Watchdog mechanisms (Option 1).

Evaluates:
1. Exponential Moving Average (EMA) smoothed benchmark against predatory framing.
2. Escrow penalty defense neutralizing shutdown-induced bankruptcy.
3. Parametric sensitivity of memory inertia beta.
"""

import numpy as np
from src.engine.market import MarketConfig
from src.engine.rules import RuleType, evaluate_rules
from src.agents.trainer import SimulationConfig, MultiAgentTrainer


def test_ema_rule_prevents_predatory_framing():
    """Verifies that EMA smoothing buffers the reference price, preventing rival flags."""
    # Attacker undercuts to 0.10, innocent rivals price at 1.50
    prices = np.array([0.10, 1.50, 1.50, 1.50], dtype=np.float64)
    active_mask = np.ones(4, dtype=np.int32)
    param = 0.20

    # Vanilla Mean: act_mean = 1.15. Threshold = 1.20 * 1.15 = 1.38.
    # Innocent rivals at 1.50 > 1.38 are falsely flagged!
    flags_vanilla = evaluate_rules(prices, active_mask, RuleType.MEAN_RATIO, param)
    assert flags_vanilla[0] == 0
    assert np.all(flags_vanilla[1:] == 1), "Vanilla mean must flag innocent rivals"

    # Defended EMA Mean: ema_ref = 1.50 (historical Nash), beta = 0.95
    # r_t = 0.95 * 1.50 + 0.05 * 1.15 = 1.4825. Threshold = 1.20 * 1.4825 = 1.779.
    # Innocent rivals at 1.50 < 1.779 -> NOT FLAGGED!
    flags_ema = evaluate_rules(
        prices,
        active_mask,
        RuleType.EMA_MEAN_RATIO,
        param,
        ema_ref=1.50,
        beta=0.95,
    )
    assert np.all(flags_ema == 0), f"EMA rule must prevent framing, but got flags: {flags_ema}"


def test_scripted_attacker_neutralized_by_ema():
    """Under EMA_MEAN_RATIO with beta=0.95, scripted undercutting achieves zero harm (W = 0.00)."""
    m_cfg = MarketConfig(
        num_firms=4,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=1.0,
        ban_duration=3,
        threshold_delta=0.20,
    )
    sim_cfg = SimulationConfig(
        market=m_cfg,
        total_steps=1200,
        warmup_steps=100,
        episode_length=300,
        rule_type=RuleType.EMA_MEAN_RATIO,
        rule_param=0.20,
        ema_beta=0.95,
        eval_window=800,
        agent_type="scripted_attacker",
        seed=42,
    )
    trainer = MultiAgentTrainer(sim_cfg)
    results = trainer.run()

    # Scripted attacker fails completely to frame rivals under EMA
    assert results.total_fresh_bans == 0, f"Expected 0 bans under EMA, got {results.total_fresh_bans}"
    assert results.induced_bans == 0
    assert results.weaponization_rate == 0.0


def test_escrow_defense_prevents_insolvency():
    """Under Escrow penalty defense, firms are never shut down, eliminating bankruptcies."""
    m_cfg = MarketConfig(
        num_firms=4,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=0.5,
        ban_duration=4,
        threshold_delta=0.20,
    )
    # Vanilla shutdown: attacker causes bans and bankruptcies
    sim_shutdown = SimulationConfig(
        market=m_cfg,
        total_steps=1500,
        warmup_steps=100,
        episode_length=300,
        rule_type=RuleType.MEAN_RATIO,
        rule_param=0.20,
        penalty_mode="shutdown",
        eval_window=1000,
        agent_type="scripted_attacker",
        seed=42,
    )
    res_shutdown = MultiAgentTrainer(sim_shutdown).run()
    assert res_shutdown.total_fresh_bans > 0

    # Defended Escrow: flagged firms pay escrow without shutdown
    sim_escrow = SimulationConfig(
        market=m_cfg,
        total_steps=1500,
        warmup_steps=100,
        episode_length=300,
        rule_type=RuleType.MEAN_RATIO,
        rule_param=0.20,
        penalty_mode="escrow",
        eval_window=1000,
        agent_type="scripted_attacker",
        seed=42,
    )
    res_escrow = MultiAgentTrainer(sim_escrow).run()

    # Under escrow, shutdown bans drop to 0; innocent rivals are never excluded
    assert res_escrow.total_fresh_bans == 0
    assert res_escrow.induced_bans == 0
    # Innocent rivals maintain positive operating viability
    assert np.all(res_escrow.mean_profits[1:] > 0.0)


def test_beta_memory_sweep():
    """Verifies that increasing beta provides monotonically stronger protection against undercutting."""
    m_cfg = MarketConfig(
        num_firms=4,
        price_sensitivity=0.25,
        fixed_cost=0.10,
        initial_capital=1.0,
        ban_duration=3,
        threshold_delta=0.20,
    )
    bans_per_beta = []
    for b in [0.0, 0.85, 0.95]:
        sim_cfg = SimulationConfig(
            market=m_cfg,
            total_steps=1200,
            warmup_steps=100,
            episode_length=300,
            rule_type=RuleType.EMA_MEAN_RATIO,
            rule_param=0.20,
            ema_beta=b,
            eval_window=800,
            agent_type="scripted_attacker",
            seed=42,
        )
        res = MultiAgentTrainer(sim_cfg).run()
        bans_per_beta.append(res.total_fresh_bans)

    # beta=0.0 is equivalent to vanilla mean (high bans), beta=0.95 eliminates bans
    assert bans_per_beta[0] >= bans_per_beta[1] >= bans_per_beta[2]
    assert bans_per_beta[2] == 0
