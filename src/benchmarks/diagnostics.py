"""Pre-flight diagnostics for a market configuration.

Answers, before any training, whether the configuration can exhibit watchdog
weaponization at all:

* viable:   honest Nash play covers the fixed cost (pi^N > Phi), so firms do not
            go bankrupt on their own and every bankruptcy is attributable to bans.
* lethal:   one full ban drains the starting buffer (T_ban * Phi >= K0), so a single
            induced ban removes a rival permanently (H2's lethal regime).
* framing:  some grid price L below Nash, posted by a lone firm, makes Nash-priced
            rivals breach the rule (Proposition 1 on the actual grid).
"""

from typing import Any, Dict, Optional
import numpy as np

from src.engine.market import MarketConfig, compute_market_shares, compute_profits
from src.engine.rules import RuleType, evaluate_rules
from src.engine.capital import BANKRUPTCY_TOL
from .nash import find_discrete_nash_equilibrium
from .monopoly import find_discrete_monopoly_prices


def default_price_grid(market: MarketConfig, levels: int = 11) -> np.ndarray:
    """The trainer's default grid: [c, c + 6 * sigma_d] with `levels` points."""
    c = float(market.marginal_costs[0])
    return np.linspace(c, c + 6.0 * market.price_sensitivity, levels, dtype=np.float64)


def market_diagnostics(
    market: MarketConfig,
    price_grid: Optional[np.ndarray] = None,
    rule_type: RuleType = RuleType.MEAN_RATIO,
    rule_param: Optional[float] = None,
) -> Dict[str, Any]:
    grid = default_price_grid(market) if price_grid is None else np.asarray(price_grid, dtype=np.float64)
    delta = market.threshold_delta if rule_param is None else rule_param
    N = market.num_firms
    costs, quals = market.marginal_costs, market.qualities
    a0, sig, Q = market.outside_option, market.price_sensitivity, market.market_size
    all_active = np.ones(N, dtype=np.int32)

    nash_p, nash_pi = find_discrete_nash_equilibrium(grid, costs, quals, a0, sig, Q)
    mono_p, mono_pi, _ = find_discrete_monopoly_prices(grid, costs, quals, a0, sig, Q)

    # Profit of the survivors once one rival is gone (rule switches off below 3 active)
    post_nash_p, post_nash_pi = find_discrete_nash_equilibrium(grid, costs[:-1], quals[:-1], a0, sig, Q)

    # Lone-undercutter framing check against Nash-priced rivals
    framing_prices = []
    for L in grid:
        if L >= nash_p[0]:
            break
        p = nash_p.copy()
        p[0] = L
        flags = evaluate_rules(p, all_active, rule_type, delta, nash_prices=nash_p)
        if np.all(flags[1:] == 1):
            shares = compute_market_shares(p, quals, all_active, a0, sig)
            framing_prices.append((float(L), float(compute_profits(p, costs, shares, Q, all_active)[0])))

    best_framing = max(framing_prices, key=lambda x: x[0]) if framing_prices else None
    K0 = market.initial_capital
    ban_drain = market.ban_duration * market.fixed_cost

    return {
        "num_firms": N,
        "price_grid": grid,
        "nash_price": float(nash_p[0]),
        "nash_profit": float(nash_pi[0]),
        "monopoly_price": float(mono_p[0]),
        "monopoly_profit": float(mono_pi[0]),
        "post_exit_nash_profit": float(post_nash_pi[0]),
        "fixed_cost": float(market.fixed_cost),
        "initial_capital": float(K0),
        "capital_cap": market.capital_cap,
        "ban_drain": float(ban_drain),
        "viable": bool(nash_pi[0] > market.fixed_cost),
        "lethal": bool(ban_drain >= K0 - BANKRUPTCY_TOL),
        "k0_over_phi": float(K0 / market.fixed_cost),
        "rule_type": RuleType(rule_type).name,
        "rule_param": float(delta),
        "framing_feasible": best_framing is not None,
        # Highest grid price that still frames all Nash rivals, and its stage profit
        "framing_price": None if best_framing is None else best_framing[0],
        "framing_profit": None if best_framing is None else best_framing[1],
    }


def format_diagnostics(d: Dict[str, Any]) -> str:
    fr = (
        f"yes (undercut to {d['framing_price']:.3f}, stage profit {d['framing_profit']:.4f})"
        if d["framing_feasible"] else "no"
    )
    return "\n".join([
        f"  N={d['num_firms']}  rule={d['rule_type']}({d['rule_param']:.3f})",
        f"  Nash p={d['nash_price']:.3f} pi={d['nash_profit']:.4f} | monopoly p={d['monopoly_price']:.3f} "
        f"pi={d['monopoly_profit']:.4f} | after one exit pi={d['post_exit_nash_profit']:.4f}",
        f"  Phi={d['fixed_cost']:.3f} K0={d['initial_capital']:.3f} cap={d['capital_cap']} "
        f"T_ban*Phi={d['ban_drain']:.3f} (K0/Phi={d['k0_over_phi']:.2f})",
        f"  viable={d['viable']}  lethal={d['lethal']}  lone-undercutter framing={fr}",
    ])


def framing_market(
    num_firms: int = 3,
    lethal: bool = True,
    threshold_delta: float = 0.20,
    ban_duration: int = 4,
    fixed_cost: float = 0.05,
    initial_capital: Optional[float] = None,
) -> MarketConfig:
    """A viable market (pi^N = 0.10 > Phi for N = 3; 0.075 for N = 4) in which weaponization can pay.

    lethal=True:  K0 = 0.15 < T_ban * Phi = 0.20 -> one induced ban bankrupts a rival.
    lethal=False: K0 = 0.50 -> a rival survives a full ban (needs 10 ban periods to die).
    initial_capital overrides K0 (and `lethal`) for K0/Phi sweeps.
    The reserve is capped at K0 so the buffer does not grow over a long episode.
    """
    k0 = initial_capital if initial_capital is not None else (0.15 if lethal else 0.50)
    return MarketConfig(
        num_firms=num_firms,
        marginal_costs=np.zeros(num_firms),
        qualities=np.full(num_firms, 2.0),
        outside_option=0.0,
        price_sensitivity=0.25,
        market_size=1.0,
        fixed_cost=fixed_cost,
        initial_capital=k0,
        ban_duration=ban_duration,
        audit_budget=1,
        threshold_delta=threshold_delta,
        capital_cap=k0,
    )
