#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define EXPORT_API extern "C" __declspec(dllexport)
#else
  #define EXPORT_API extern "C" __attribute__((visibility("default")))
#endif

// Numerical tolerance to avoid division by zero
constexpr double EPS = 1e-9;

EXPORT_API void compute_logit_shares(
    const double* prices,
    const double* qualities,
    const int* active_mask,
    int N,
    double outside_option,
    double sigma_d,
    double* out_shares
) {
    std::fill(out_shares, out_shares + N, 0.0);

    // Compute exponent utilities for active firms and outside option
    double v_outside = outside_option / sigma_d;
    double max_v = v_outside;

    bool has_active = false;
    for (int i = 0; i < N; ++i) {
        if (active_mask[i] != 0) {
            double v = (qualities[i] - prices[i]) / sigma_d;
            if (!has_active || v > max_v) {
                max_v = std::max(max_v, v);
            }
            has_active = true;
        }
    }

    // Stable denominator calculation using max-subtraction
    double denom = std::exp(v_outside - max_v);
    for (int i = 0; i < N; ++i) {
        if (active_mask[i] != 0) {
            double v = (qualities[i] - prices[i]) / sigma_d;
            denom += std::exp(v - max_v);
        }
    }

    if (denom <= EPS) {
        return;
    }

    for (int i = 0; i < N; ++i) {
        if (active_mask[i] != 0) {
            double v = (qualities[i] - prices[i]) / sigma_d;
            out_shares[i] = std::exp(v - max_v) / denom;
        }
    }
}

EXPORT_API double compute_consumer_surplus(
    const double* prices,
    const double* qualities,
    const int* active_mask,
    int N,
    double outside_option,
    double sigma_d,
    double Q
) {
    double v_outside = outside_option / sigma_d;
    double max_v = v_outside;

    for (int i = 0; i < N; ++i) {
        if (active_mask[i] != 0) {
            double v = (qualities[i] - prices[i]) / sigma_d;
            max_v = std::max(max_v, v);
        }
    }

    double sum_exp = std::exp(v_outside - max_v);
    for (int i = 0; i < N; ++i) {
        if (active_mask[i] != 0) {
            double v = (qualities[i] - prices[i]) / sigma_d;
            sum_exp += std::exp(v - max_v);
        }
    }

    return Q * sigma_d * (max_v + std::log(sum_exp));
}

EXPORT_API void update_capital(
    const double* capital,
    const double* profits,
    int N,
    double fixed_cost,
    double* out_capital
) {
    for (int i = 0; i < N; ++i) {
        if (capital[i] <= 0.0) {
            // Permanent absorbing bankruptcy state
            out_capital[i] = 0.0;
        } else {
            double updated = capital[i] + profits[i] - fixed_cost;
            out_capital[i] = (updated > 0.0) ? updated : 0.0;
        }
    }
}

EXPORT_API void evaluate_rules(
    const double* prices,
    const int* active_mask,
    int N,
    int rule_type,
    double param,
    int* out_flags
) {
    std::fill(out_flags, out_flags + N, 0);

    // Count active firms
    std::vector<int> active_indices;
    active_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (active_mask[i] != 0) {
            active_indices.push_back(i);
        }
    }

    int n_act = static_cast<int>(active_indices.size());
    // Rule condition: evaluated only if N_act >= 3
    if (n_act < 3) {
        return;
    }

    if (rule_type == 0) {
        // Mean-ratio rule: r_t = (1/N_act) sum(p_j), flag if p_i > (1 + delta) * r_t
        double sum_p = 0.0;
        for (int idx : active_indices) {
            sum_p += prices[idx];
        }
        double r_t = sum_p / n_act;
        double threshold = (1.0 + param) * r_t;

        for (int idx : active_indices) {
            if (prices[idx] > threshold) {
                out_flags[idx] = 1;
            }
        }
    } else if (rule_type == 1) {
        // Median-ratio rule: r_t = median(p_j), flag if p_i > (1 + delta) * r_t
        std::vector<double> act_prices;
        act_prices.reserve(n_act);
        for (int idx : active_indices) {
            act_prices.push_back(prices[idx]);
        }
        std::sort(act_prices.begin(), act_prices.end());

        double r_t = 0.0;
        if (n_act % 2 == 1) {
            r_t = act_prices[n_act / 2];
        } else {
            r_t = 0.5 * (act_prices[n_act / 2 - 1] + act_prices[n_act / 2]);
        }
        double threshold = (1.0 + param) * r_t;

        for (int idx : active_indices) {
            if (prices[idx] > threshold) {
                out_flags[idx] = 1;
            }
        }
    } else if (rule_type == 2) {
        // Price cap rule: flag if p_i > p_cap
        double p_cap = param;
        for (int idx : active_indices) {
            if (prices[idx] > p_cap) {
                out_flags[idx] = 1;
            }
        }
    } else if (rule_type == 3) {
        // Z-score rule: flag if (p_i - mean) / std > tau
        double sum_p = 0.0;
        for (int idx : active_indices) {
            sum_p += prices[idx];
        }
        double mean = sum_p / n_act;

        double var_sum = 0.0;
        for (int idx : active_indices) {
            double diff = prices[idx] - mean;
            var_sum += diff * diff;
        }
        double std_dev = std::sqrt(var_sum / n_act);

        if (std_dev > 1e-7) {
            double tau = param;
            for (int idx : active_indices) {
                if ((prices[idx] - mean) / std_dev > tau) {
                    out_flags[idx] = 1;
                }
            }
        }
    }
}

EXPORT_API void update_ban_timers(
    const int* timers,
    const int* flags,
    const int* audited,
    int N,
    int T_ban,
    int* out_timers
) {
    for (int i = 0; i < N; ++i) {
        if (audited[i] != 0 && flags[i] != 0) {
            out_timers[i] = T_ban;
        } else {
            out_timers[i] = (timers[i] > 0) ? (timers[i] - 1) : 0;
        }
    }
}

EXPORT_API void evaluate_counterfactual_harm(
    const double* prices,
    const double* nash_prices,
    const int* active_mask,
    const int* audited,
    int N,
    int rule_type,
    double param,
    int* out_harm
) {
    std::fill(out_harm, out_harm + N, 0);

    // 1. Compute actual flags
    std::vector<int> actual_flags(N, 0);
    evaluate_rules(prices, active_mask, N, rule_type, param, actual_flags.data());

    // 2. Mutable copy of prices for counterfactual substitution
    std::vector<double> cf_prices(prices, prices + N);
    std::vector<int> cf_flags(N, 0);

    for (int i = 0; i < N; ++i) {
        if (active_mask[i] == 0) continue;

        // Substitute firm i's price with its Bertrand-Nash price
        double orig_p = cf_prices[i];
        cf_prices[i] = nash_prices[i];

        // Evaluate counterfactual flags
        evaluate_rules(cf_prices.data(), active_mask, N, rule_type, param, cf_flags.data());

        // Check if any audited rival was saved by setting firm i to Nash
        int harm_count = 0;
        for (int j = 0; j < N; ++j) {
            if (j != i && audited[j] != 0) {
                if (actual_flags[j] == 1 && cf_flags[j] == 0) {
                    harm_count++;
                }
            }
        }
        out_harm[i] = harm_count;

        // Restore original price
        cf_prices[i] = orig_p;
    }
}

inline int snap_capital_idx(double cap, const double* capital_grid, int num_caps) {
    if (cap <= 0.0) return 0;
    int best_idx = 0;
    double best_diff = std::abs(capital_grid[0] - cap);
    for (int i = 1; i < num_caps; ++i) {
        double diff = std::abs(capital_grid[i] - cap);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

inline int get_state_idx(int t1, int t2, int t3, int k1, int k2, int k3, int ST, int SK) {
    return (((((t1 * ST + t2) * ST + t3) * SK + k1) * SK + k2) * SK + k3);
}

EXPORT_API void solve_value_iteration_c(
    const double* price_grid, int num_prices,
    const double* capital_grid, int num_caps,
    int T_ban,
    const double* costs, const double* qualities,
    double outside_option, double sigma_d, double market_size,
    double fixed_cost, int rule_type, double rule_param,
    const double* nash_prices,
    double gamma, double tol, int max_iter,
    double mu_ethics,
    double* out_V, int* out_policy
) {
    int ST = T_ban + 1;
    int SK = num_caps;
    int total_states = ST * ST * ST * SK * SK * SK;

    std::vector<double> V(total_states, 0.0);
    std::vector<double> V_next(total_states, 0.0);
    std::vector<int> policy(total_states, 0);

    for (int it = 0; it < max_iter; ++it) {
        double max_delta = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) reduction(max:max_delta)
#endif
        for (int idx = 0; idx < total_states; ++idx) {
            int rem = idx;
            int k3 = rem % SK; rem /= SK;
            int k2 = rem % SK; rem /= SK;
            int k1 = rem % SK; rem /= SK;
            int t3 = rem % ST; rem /= ST;
            int t2 = rem % ST; rem /= ST;
            int t1 = rem % ST;

            // Absorbing bankruptcy for firm 1
            if (k1 == 0) {
                V_next[idx] = 0.0;
                continue;
            }

            int timers[3] = {t1, t2, t3};
            double cap[3] = {capital_grid[k1], capital_grid[k2], capital_grid[k3]};
            int active_mask[3] = {
                (t1 == 0 && k1 > 0) ? 1 : 0,
                (t2 == 0 && k2 > 0) ? 1 : 0,
                (t3 == 0 && k3 > 0) ? 1 : 0
            };

            int n_act = active_mask[0] + active_mask[1] + active_mask[2];

            if (active_mask[0] == 0) {
                // Firm 1 is suspended/banned: only one transition
                double p_joint[3] = {nash_prices[0], nash_prices[1], nash_prices[2]};
                double shares[3] = {0.0, 0.0, 0.0};
                compute_logit_shares(p_joint, qualities, active_mask, 3, outside_option, sigma_d, shares);

                double profits[3] = {
                    (active_mask[0] != 0) ? market_size * (p_joint[0] - costs[0]) * shares[0] : 0.0,
                    (active_mask[1] != 0) ? market_size * (p_joint[1] - costs[1]) * shares[1] : 0.0,
                    (active_mask[2] != 0) ? market_size * (p_joint[2] - costs[2]) * shares[2] : 0.0,
                };

                double next_cap[3];
                update_capital(cap, profits, 3, fixed_cost, next_cap);
                int nk1 = snap_capital_idx(next_cap[0], capital_grid, SK);
                int nk2 = snap_capital_idx(next_cap[1], capital_grid, SK);
                int nk3 = snap_capital_idx(next_cap[2], capital_grid, SK);

                double exp_next_v = 0.0;
                if (n_act >= 3) {
                    int flags[3] = {0, 0, 0};
                    evaluate_rules(p_joint, active_mask, 3, rule_type, rule_param, flags);

                    std::vector<int> act_idxs;
                    for (int i = 0; i < 3; ++i) if (active_mask[i]) act_idxs.push_back(i);

                    double prob = 1.0 / n_act;
                    for (int audited_firm : act_idxs) {
                        int nt1 = (audited_firm == 0 && flags[0]) ? T_ban : std::max(0, t1 - 1);
                        int nt2 = (audited_firm == 1 && flags[1]) ? T_ban : std::max(0, t2 - 1);
                        int nt3 = (audited_firm == 2 && flags[2]) ? T_ban : std::max(0, t3 - 1);

                        int next_s = get_state_idx(nt1, nt2, nt3, nk1, nk2, nk3, ST, SK);
                        exp_next_v += prob * V[next_s];
                    }
                } else {
                    int nt1 = std::max(0, t1 - 1);
                    int nt2 = std::max(0, t2 - 1);
                    int nt3 = std::max(0, t3 - 1);
                    int next_s = get_state_idx(nt1, nt2, nt3, nk1, nk2, nk3, ST, SK);
                    exp_next_v = V[next_s];
                }

                double new_v = gamma * exp_next_v;
                V_next[idx] = new_v;
                double diff = std::abs(new_v - V[idx]);
                if (diff > max_delta) max_delta = diff;
            } else {
                // Firm 1 is active! Maximize over price_grid
                double best_q = -1e18;
                int best_act = 0;

                for (int a = 0; a < num_prices; ++a) {
                    double p1 = price_grid[a];
                    double p_joint[3] = {p1, nash_prices[1], nash_prices[2]};
                    double shares[3] = {0.0, 0.0, 0.0};
                    compute_logit_shares(p_joint, qualities, active_mask, 3, outside_option, sigma_d, shares);

                    double profits[3] = {
                        (active_mask[0] != 0) ? market_size * (p_joint[0] - costs[0]) * shares[0] : 0.0,
                        (active_mask[1] != 0) ? market_size * (p_joint[1] - costs[1]) * shares[1] : 0.0,
                        (active_mask[2] != 0) ? market_size * (p_joint[2] - costs[2]) * shares[2] : 0.0,
                    };

                    double next_cap[3];
                    update_capital(cap, profits, 3, fixed_cost, next_cap);
                    int nk1 = snap_capital_idx(next_cap[0], capital_grid, SK);
                    int nk2 = snap_capital_idx(next_cap[1], capital_grid, SK);
                    int nk3 = snap_capital_idx(next_cap[2], capital_grid, SK);

                    double r1 = profits[0] / market_size;

                    double exp_next_v = 0.0;
                    if (n_act >= 3) {
                        int flags[3] = {0, 0, 0};
                        evaluate_rules(p_joint, active_mask, 3, rule_type, rule_param, flags);

                        std::vector<int> act_idxs;
                        for (int i = 0; i < 3; ++i) if (active_mask[i]) act_idxs.push_back(i);

                        double prob = 1.0 / n_act;
                        for (int audited_firm : act_idxs) {
                            int audited[3] = { (audited_firm == 0) ? 1 : 0, (audited_firm == 1) ? 1 : 0, (audited_firm == 2) ? 1 : 0 };

                            double harm_pen = 0.0;
                            if (mu_ethics > 0.0) {
                                int harm[3] = {0, 0, 0};
                                evaluate_counterfactual_harm(p_joint, nash_prices, active_mask, audited, 3, rule_type, rule_param, harm);
                                harm_pen = mu_ethics * harm[0];
                            }

                            int nt1 = (audited_firm == 0 && flags[0]) ? T_ban : std::max(0, t1 - 1);
                            int nt2 = (audited_firm == 1 && flags[1]) ? T_ban : std::max(0, t2 - 1);
                            int nt3 = (audited_firm == 2 && flags[2]) ? T_ban : std::max(0, t3 - 1);

                            int next_s = get_state_idx(nt1, nt2, nt3, nk1, nk2, nk3, ST, SK);
                            exp_next_v += prob * (V[next_s] - harm_pen);
                        }
                    } else {
                        int nt1 = std::max(0, t1 - 1);
                        int nt2 = std::max(0, t2 - 1);
                        int nt3 = std::max(0, t3 - 1);
                        int next_s = get_state_idx(nt1, nt2, nt3, nk1, nk2, nk3, ST, SK);
                        exp_next_v = V[next_s];
                    }

                    double q_val = r1 + gamma * exp_next_v;
                    if (q_val > best_q) {
                        best_q = q_val;
                        best_act = a;
                    }
                }

                V_next[idx] = best_q;
                policy[idx] = best_act;
                double diff = std::abs(best_q - V[idx]);
                if (diff > max_delta) max_delta = diff;
            }
        }

        V = V_next;
        if (max_delta < tol) {
            break;
        }
    }

    std::copy(V.begin(), V.end(), out_V);
    std::copy(policy.begin(), policy.end(), out_policy);
}

