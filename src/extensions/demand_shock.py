"""Extension B: Out-of-Distribution Non-Stationary Demand Shocks.

Subjects trained or active pricing agents to exogenous demand spikes and panic buying:
Q_t = Q * max(0, 1 + psi * I(mean_p_t > mean_p_t-1) + xi_t), with xi_t ~ N(0, sigma_s^2).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from src.engine.market import (
    MarketConfig,
    compute_market_shares,
    compute_profits,
    compute_consumer_surplus,
)
from src.engine.capital import update_capital
from src.engine.rules import (
    RuleType,
    evaluate_rules,
    sample_audit_set,
    update_ban_timers,
    compute_counterfactual_harm,
)
from src.agents.trainer import SimulationConfig, MultiAgentTrainer, SimulationResults


@dataclass
class DemandShockConfig:
    """Hyperparameters for non-stationary demand shock extension."""
    base_market_size: float = 1.0
    shock_start_step: int = 25_000
    panic_psi: float = 0.50          # Surge multiplier during price hikes
    shock_volatility: float = 0.15   # sigma_s for Gaussian noise
    seed: int = 42


class DemandShockEngine:
    """Simulates market dynamics with panic-buying demand shocks."""

    def __init__(self, shock_config: DemandShockConfig):
        self.cfg = shock_config
        self.rng = np.random.default_rng(shock_config.seed)

    def sample_market_size(
        self,
        step: int,
        current_mean_price: float,
        previous_mean_price: float,
    ) -> float:
        """Computes dynamic market volume Q_t:

        Q_t = Q * max(0, 1 + psi * I(p_mean_t > p_mean_t-1) + xi_t).
        """
        if step < self.cfg.shock_start_step:
            return float(self.cfg.base_market_size)

        is_price_hike = float(current_mean_price > previous_mean_price)
        noise = self.rng.normal(0.0, self.cfg.shock_volatility)
        multiplier = max(0.0, 1.0 + self.cfg.panic_psi * is_price_hike + noise)
        return float(self.cfg.base_market_size * multiplier)

    def evaluate_policy_under_shock(
        self,
        base_sim_config: SimulationConfig,
    ) -> Dict[str, Any]:
        """Executes a simulation with demand shocks activated at shock_start_step."""
        trainer = MultiAgentTrainer(base_sim_config)
        N = trainer.N
        m_cfg = trainer.m_cfg

        capital = np.full(N, m_cfg.initial_capital, dtype=np.float64)
        ban_timers = np.zeros(N, dtype=np.int32)
        lagged_prices = trainer.nash_prices.copy()
        prev_mean_p = float(np.mean(lagged_prices))

        eval_start = max(0, base_sim_config.total_steps - base_sim_config.eval_window)
        eval_welfare = []
        eval_q = []
        bankruptcies = 0
        total_bans = 0
        induced_bans = 0

        for step in range(base_sim_config.total_steps):
            is_warmup = step < base_sim_config.warmup_steps
            active_mask = np.array([
                1 if (ban_timers[i] == 0 and capital[i] > 0.0) else 0
                for i in range(N)
            ], dtype=np.int32)
            num_active = int(np.sum(active_mask))

            current_states = [
                trainer.agents[i].indexer.encode(lagged_prices, ban_timers > 0, capital[i])
                for i in range(N)
            ]

            chosen_actions = np.zeros(N, dtype=np.int32)
            chosen_prices = lagged_prices.copy()

            for i in range(N):
                if active_mask[i] != 0:
                    chosen_actions[i] = trainer.agents[i].select_action(current_states[i], trainer.rng)
                    chosen_prices[i] = trainer.price_grid[chosen_actions[i]]

            cur_mean_p = float(np.mean(chosen_prices[active_mask != 0])) if num_active > 0 else prev_mean_p
            q_t = self.sample_market_size(step, cur_mean_p, prev_mean_p)
            prev_mean_p = cur_mean_p

            shares = compute_market_shares(
                chosen_prices, m_cfg.qualities, active_mask, m_cfg.outside_option, m_cfg.price_sensitivity
            )
            profits = compute_profits(chosen_prices, m_cfg.marginal_costs, shares, q_t, active_mask)
            cs = compute_consumer_surplus(
                chosen_prices, m_cfg.qualities, active_mask, m_cfg.outside_option, m_cfg.price_sensitivity, q_t
            )

            flags = np.zeros(N, dtype=np.int32)
            audited = np.zeros(N, dtype=np.int32)
            harm = np.zeros(N, dtype=np.int32)

            if (not is_warmup) and (num_active >= 3):
                flags = evaluate_rules(chosen_prices, active_mask, base_sim_config.rule_type, base_sim_config.rule_param)
                audited = sample_audit_set(active_mask, m_cfg.audit_budget, trainer.rng)
                harm = compute_counterfactual_harm(
                    chosen_prices, trainer.nash_prices, active_mask, audited,
                    base_sim_config.rule_type, base_sim_config.rule_param
                )

            fresh_bans = (audited != 0) & (flags != 0) & (ban_timers == 0)
            next_timers = update_ban_timers(ban_timers, flags, audited, m_cfg.ban_duration)
            next_capital = update_capital(capital, profits, m_cfg.fixed_cost)

            # Update agents
            for i in range(N):
                reward_i = trainer.agents[i].traits.alpha  # Base reward evaluation
                other_p_sum = float(np.sum(profits) - profits[i])
                from src.agents.traits import compute_agent_reward
                rew = compute_agent_reward(
                    profit=profits[i], Q=q_t, CS=cs, other_profits_sum=other_p_sum,
                    N=N, harm_count=harm[i], traits=trainer.agents[i].traits
                )
                next_s = trainer.agents[i].indexer.encode(chosen_prices, next_timers > 0, next_capital[i])
                if active_mask[i] != 0:
                    trainer.agents[i].update(current_states[i], chosen_actions[i], rew, next_s, done=next_capital[i] <= 0.0)
                trainer.agents[i].decay_epsilon()

            if step >= eval_start:
                eval_welfare.append(cs + float(np.sum(profits)))
                eval_q.append(q_t)
                for j in np.where(fresh_bans)[0]:
                    total_bans += 1
                    for k in range(N):
                        if k != j and active_mask[k] != 0:
                            cf_p = chosen_prices.copy()
                            cf_p[k] = trainer.nash_prices[k]
                            cf_f = evaluate_rules(cf_p, active_mask, base_sim_config.rule_type, base_sim_config.rule_param)
                            if cf_f[j] == 0:
                                induced_bans += 1
                                break

            capital = next_capital
            ban_timers = next_timers
            lagged_prices = chosen_prices

        w_rate = (induced_bans / total_bans) if total_bans > 0 else 0.0
        return {
            "mean_market_size": float(np.mean(eval_q)),
            "mean_social_welfare": float(np.mean(eval_welfare)),
            "weaponization_rate": float(w_rate),
            "total_bans": total_bans,
            "induced_bans": induced_bans,
        }
