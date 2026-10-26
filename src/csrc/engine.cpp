#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>

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
