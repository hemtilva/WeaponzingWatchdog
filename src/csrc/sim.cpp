// Native simulation loop: regulated logit oligopoly + tabular Q-learning sellers.
//
// Mirrors src/agents/trainer.py::MultiAgentTrainer.run step for step (that file is
// the readable reference; tests/test_native_sim.py checks agreement). Q-tables and
// visit counts live in caller-owned (NumPy) memory so they persist across calls and
// stay inspectable from Python.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

#if defined(_WIN32) || defined(__CYGWIN__)
  #define SIM_API extern "C" __declspec(dllexport)
#else
  #define SIM_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

enum Policy { POL_Q = 0, POL_NASH = 1, POL_RANDOM = 2, POL_ATTACKER = 3 };
enum Rule { R_MEAN = 0, R_MEDIAN = 1, R_CAP = 2, R_ZSCORE = 3, R_EMA = 4 };
constexpr double BANKRUPTCY_TOL = 1e-9;

// ---------------------------------------------------------------- RNG (xoshiro256**)
struct Rng {
    uint64_t s[4];
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    explicit Rng(uint64_t seed) { for (auto& v : s) v = splitmix(seed); }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return result;
    }
    double uniform() { return (next() >> 11) * 0x1.0p-53; }
    int below(int n) { return static_cast<int>(uniform() * n); }
};

}  // namespace

// Layout must match NativeSimParams in src/engine/native_sim.py (field order and types).
struct SimParams {
    int32_t N, M;
    int32_t rule_type, penalty_mode;          // penalty_mode: 0 shutdown, 1 escrow
    int32_t total_steps, warmup_steps, episode_length, eval_window;
    int32_t audit_budget, T_ban, freeze, has_cap;
    double outside, sigma, Q, fixed_cost, K0, cap;
    double rule_param, ema_beta;
    double lr, gamma, eps_min, eps_decay, reward_eps;
    uint64_t seed;
    const double* price_grid;     // (M)
    const double* costs;          // (N)
    const double* qualities;      // (N)
    const double* nash_prices;    // (N)
    const double* nash_profits;   // (N)
    const int32_t* nash_idx;      // (N)
    const int32_t* mono_idx;      // (N)
    const int32_t* policies;      // (N)
    const double* alpha;          // (N)
    const double* lambda_;        // (N)
    const double* mu;             // (N)
    float* qtab;                  // (N, S, M), S = M^N * 2^N * 3
    int32_t* visits;              // (N, S)
    double* epsilon;              // (N) in/out
};

struct SimOutput {
    double* mean_prices;          // (N) over active eval periods (Nash price if never active)
    double* mean_profits;         // (N)
    int64_t* induced_by;          // (N)
    int64_t* kills_by;            // (N)
    // counts: fresh_bans, induced_bans, sanctions, induced_sanctions,
    //         bankruptcies, induced_bankruptcies, episodes_in_eval, eval_periods
    int64_t* counts;              // (8)
    double* mean_cs;              // (1)
};

namespace {

struct Sim {
    const SimParams& P;
    int N, M;
    int64_t S, num_bans;
    Rng rng;

    Sim(const SimParams& p) : P(p), N(p.N), M(p.M), rng(p.seed) {
        num_bans = int64_t(1) << N;
        int64_t pc = 1;
        for (int i = 0; i < N; ++i) pc *= M;
        S = pc * num_bans * 3;
    }

    int cap_bin(double k) const {
        if (k <= 2.0 * P.fixed_cost) return 0;
        if (k <= 5.0 * P.fixed_cost) return 1;
        return 2;
    }

    int64_t encode(const int* lag_idx, const int* inactive, double k) const {
        int64_t pc = 0;
        for (int f = 0; f < N; ++f) pc = pc * M + lag_idx[f];
        int64_t bc = 0;
        for (int f = 0; f < N; ++f) bc = (bc << 1) | (inactive[f] ? 1 : 0);
        return (pc * num_bans + bc) * 3 + cap_bin(k);
    }

    float* qrow(int i, int64_t s) const { return P.qtab + (int64_t(i) * S + s) * M; }

    double crra(double x, double a) const {
        if (std::fabs(a - 1.0) < 1e-6) return std::log(x);
        return (std::pow(x, 1.0 - a) - 1.0) / (1.0 - a);
    }

    // Regulatory flags (all-zero below 3 active firms)
    void flags(const double* p, const int* act, double ema_ref, int* out) const {
        std::fill(out, out + N, 0);
        double ap[16];
        int idx[16];
        int n = 0;
        for (int i = 0; i < N; ++i) if (act[i]) { ap[n] = p[i]; idx[n] = i; ++n; }
        if (n < 3) return;
        double mean = 0.0;
        for (int k = 0; k < n; ++k) mean += ap[k];
        mean /= n;
        const double d = P.rule_param;
        switch (P.rule_type) {
            case R_MEAN:
                for (int k = 0; k < n; ++k) if (ap[k] > (1.0 + d) * mean) out[idx[k]] = 1;
                break;
            case R_EMA: {
                double r = (ema_ref > 0.0) ? P.ema_beta * ema_ref + (1.0 - P.ema_beta) * mean : mean;
                for (int k = 0; k < n; ++k) if (ap[k] > (1.0 + d) * r) out[idx[k]] = 1;
                break;
            }
            case R_MEDIAN: {
                double sorted[16];
                std::copy(ap, ap + n, sorted);
                std::sort(sorted, sorted + n);
                double med = (n % 2) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
                for (int k = 0; k < n; ++k) if (ap[k] > (1.0 + d) * med) out[idx[k]] = 1;
                break;
            }
            case R_CAP:
                for (int k = 0; k < n; ++k) if (ap[k] > d) out[idx[k]] = 1;
                break;
            case R_ZSCORE: {
                double var = 0.0;
                for (int k = 0; k < n; ++k) var += (ap[k] - mean) * (ap[k] - mean);
                double sd = std::sqrt(var / n);
                if (sd > 1e-7)
                    for (int k = 0; k < n; ++k) if ((ap[k] - mean) / sd > d) out[idx[k]] = 1;
                break;
            }
        }
    }

    int select(int i, int64_t s, int num_active) {
        switch (P.policies[i]) {
            case POL_NASH: return P.nash_idx[i];
            case POL_RANDOM: return rng.below(M);
            case POL_ATTACKER: return (num_active >= 3) ? 0 : P.mono_idx[i];
            default: break;
        }
        if (!P.freeze && rng.uniform() < P.epsilon[i]) return rng.below(M);
        const float* q = qrow(i, s);
        float mx = q[0];
        for (int a = 1; a < M; ++a) mx = std::max(mx, q[a]);
        int best[64], nb = 0;
        for (int a = 0; a < M; ++a) if (std::fabs(q[a] - mx) < 1e-6f) best[nb++] = a;
        return nb == 1 ? best[0] : best[rng.below(nb)];
    }
};

struct Pending {
    bool open = false;
    int64_t s = 0;
    int a = 0;
    double acc = 0.0, disc = 1.0;
};

}  // namespace

SIM_API int64_t sim_num_states(int32_t N, int32_t M) {
    int64_t pc = 1;
    for (int i = 0; i < N; ++i) pc *= M;
    return pc * (int64_t(1) << N) * 3;
}

SIM_API int32_t run_simulation_c(const SimParams* params, SimOutput* out) {
    const SimParams& P = *params;
    if (P.N < 2 || P.N > 16 || P.M < 2 || P.M > 64) return -1;
    Sim sim(P);
    const int N = P.N;

    auto close = [&](int i, Pending& pd, int64_t next_s, bool terminal) {
        if (!pd.open) return;
        pd.open = false;
        P.visits[int64_t(i) * sim.S + pd.s] += 1;
        double target = pd.acc;
        if (!terminal) {
            const float* q = sim.qrow(i, next_s);
            float mx = q[0];
            for (int a = 1; a < P.M; ++a) mx = std::max(mx, q[a]);
            target += pd.disc * mx;
        }
        float* cell = sim.qrow(i, pd.s) + pd.a;
        *cell += float(P.lr * (target - *cell));
    };

    const double cap0 = P.has_cap ? std::min(P.K0, P.cap) : P.K0;
    std::vector<double> capital(N, cap0), next_capital(N), prices(N), profits(N), shares(N);
    std::vector<int> timers(N, 0), next_timers(N), lag_idx(N), act_idx(N), active(N), inactive(N);
    std::vector<int> flg(N), cf_flg(N), audited(N), harm(N), sanctioned(N), sanction_inducer(N), ban_inducer(N, -1);
    std::vector<int64_t> states(N);
    std::vector<Pending> pend(N);
    std::vector<double> price_sum(N, 0.0), profit_sum(N, 0.0);
    std::vector<int64_t> price_cnt(N, 0);
    std::vector<int> act_list(N);
    int64_t fresh_bans = 0, induced_bans = 0, sanctions = 0, induced_sanctions = 0;
    int64_t bankruptcies = 0, induced_bankruptcies = 0, episodes_in_eval = 0, eval_periods = 0;
    double cs_sum = 0.0;
    for (int i = 0; i < N; ++i) { out->induced_by[i] = 0; out->kills_by[i] = 0; }

    auto reset = [&](double& ema) {
        for (int i = 0; i < N; ++i) {
            capital[i] = cap0; timers[i] = 0; lag_idx[i] = P.nash_idx[i]; ban_inducer[i] = -1;
        }
        double m = 0.0;
        for (int i = 0; i < N; ++i) m += P.price_grid[P.nash_idx[i]];
        ema = m / N;
    };
    double ema_ref = 0.0;
    reset(ema_ref);

    const int eval_start = std::max(0, P.total_steps - P.eval_window);
    const double v_out = P.outside / P.sigma;
    int episode_step = 0;

    for (int step = 0; step < P.total_steps; ++step) {
        const bool in_eval = step >= eval_start;
        if (step == eval_start) ++episodes_in_eval;
        const bool warmup = step < P.warmup_steps;
        int n_act = 0;
        for (int i = 0; i < N; ++i) {
            active[i] = (timers[i] == 0 && capital[i] > 0.0) ? 1 : 0;
            inactive[i] = (timers[i] > 0 || capital[i] <= 0.0) ? 1 : 0;
            n_act += active[i];
        }

        // 1. States; close transitions of learners acting again
        for (int i = 0; i < N; ++i) states[i] = sim.encode(lag_idx.data(), inactive.data(), capital[i]);
        for (int i = 0; i < N; ++i)
            if (P.policies[i] == POL_Q && active[i] && pend[i].open) close(i, pend[i], states[i], false);

        // 2. Actions (inactive firms keep their last price)
        for (int i = 0; i < N; ++i) {
            if (active[i]) {
                int a = sim.select(i, states[i], n_act);
                act_idx[i] = a;
                if (P.policies[i] == POL_Q && !P.freeze) {
                    pend[i].open = true; pend[i].s = states[i]; pend[i].a = a;
                    pend[i].acc = 0.0; pend[i].disc = 1.0;
                }
            } else {
                act_idx[i] = lag_idx[i];
            }
            prices[i] = P.price_grid[act_idx[i]];
        }

        // 3. Market clearing
        double max_v = v_out;
        for (int i = 0; i < N; ++i)
            if (active[i]) max_v = std::max(max_v, (P.qualities[i] - prices[i]) / P.sigma);
        double denom = std::exp(v_out - max_v);
        for (int i = 0; i < N; ++i)
            if (active[i]) denom += std::exp((P.qualities[i] - prices[i]) / P.sigma - max_v);
        double total_profit = 0.0;
        for (int i = 0; i < N; ++i) {
            shares[i] = active[i] ? std::exp((P.qualities[i] - prices[i]) / P.sigma - max_v) / denom : 0.0;
            profits[i] = active[i] ? P.Q * (prices[i] - P.costs[i]) * shares[i] : 0.0;
        }
        const double cs = P.Q * P.sigma * (max_v + std::log(denom));

        // 4. Regulation
        std::fill(flg.begin(), flg.end(), 0);
        std::fill(audited.begin(), audited.end(), 0);
        std::fill(harm.begin(), harm.end(), 0);
        const double ema_used = ema_ref;
        const bool regulated = !warmup && n_act >= 3;
        if (regulated) {
            sim.flags(prices.data(), active.data(), ema_used, flg.data());
            int na = 0;
            for (int i = 0; i < N; ++i) if (active[i]) act_list[na++] = i;
            int b = std::min(P.audit_budget, na);
            for (int k = 0; k < b; ++k) {  // partial Fisher-Yates: sample without replacement
                int r = k + sim.rng.below(na - k);
                std::swap(act_list[k], act_list[r]);
                audited[act_list[k]] = 1;
            }
            // Counterfactual harm: i's deviation from Nash caused audited rival j's flag
            for (int i = 0; i < N; ++i) {
                if (!active[i]) continue;
                double keep = prices[i];
                prices[i] = P.nash_prices[i];
                sim.flags(prices.data(), active.data(), ema_used, cf_flg.data());
                prices[i] = keep;
                for (int j = 0; j < N; ++j)
                    if (j != i && audited[j] && flg[j] && !cf_flg[j]) ++harm[i];
            }
        }
        if (n_act > 0) {
            double m = 0.0;
            for (int i = 0; i < N; ++i) if (active[i]) m += prices[i];
            ema_ref = P.ema_beta * ema_ref + (1.0 - P.ema_beta) * (m / n_act);
        }

        // Sanctions and their inducers
        for (int j = 0; j < N; ++j) {
            sanctioned[j] = (audited[j] && flg[j] && timers[j] == 0) ? 1 : 0;
            sanction_inducer[j] = -1;
            if (!sanctioned[j]) continue;
            double best_gap = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < N; ++k) {
                if (k == j || !active[k]) continue;
                double keep = prices[k];
                prices[k] = P.nash_prices[k];
                sim.flags(prices.data(), active.data(), ema_used, cf_flg.data());
                prices[k] = keep;
                if (!cf_flg[j]) {
                    double gap = P.nash_prices[k] - prices[k];
                    if (gap > best_gap) { best_gap = gap; sanction_inducer[j] = k; }
                }
            }
        }

        if (P.penalty_mode == 1) {  // escrow
            for (int i = 0; i < N; ++i)
                if (audited[i] && flg[i]) profits[i] -= std::max(0.0, profits[i] - P.nash_profits[i]);
            for (int i = 0; i < N; ++i) next_timers[i] = timers[i];
        } else {
            for (int i = 0; i < N; ++i) {
                next_timers[i] = (audited[i] && flg[i]) ? P.T_ban : std::max(0, timers[i] - 1);
                if (sanctioned[i]) ban_inducer[i] = sanction_inducer[i];
            }
        }
        total_profit = 0.0;
        for (int i = 0; i < N; ++i) total_profit += profits[i];

        // 5. Capital
        for (int i = 0; i < N; ++i) {
            double k = capital[i] > 0.0 ? std::max(0.0, capital[i] + profits[i] - P.fixed_cost) : 0.0;
            if (k <= BANKRUPTCY_TOL) k = 0.0;
            if (P.has_cap) k = std::min(k, P.cap);
            next_capital[i] = k;
        }

        // 6. Rewards & learning
        for (int i = 0; i < N; ++i) {
            if (P.policies[i] != POL_Q) continue;
            if (pend[i].open) {
                const double a = P.alpha[i];
                double x = std::max(0.0, profits[i]) / P.Q + P.reward_eps;
                double r = sim.crra(x, a) - sim.crra(P.reward_eps, a);
                r += P.lambda_[i] * (cs + total_profit - profits[i]) / (N * P.Q);
                r -= P.mu[i] * harm[i];
                pend[i].acc += pend[i].disc * r;
                pend[i].disc *= P.gamma;
                if (capital[i] > 0.0 && next_capital[i] <= 0.0) close(i, pend[i], 0, true);
            }
            if (!P.freeze) P.epsilon[i] = std::max(P.eps_min, P.epsilon[i] * P.eps_decay);
        }

        // Evaluation bookkeeping
        if (in_eval) {
            ++eval_periods;
            cs_sum += cs;
            for (int i = 0; i < N; ++i) {
                profit_sum[i] += profits[i];
                if (active[i]) { price_sum[i] += prices[i]; ++price_cnt[i]; }
            }
            for (int j = 0; j < N; ++j) {
                if (!sanctioned[j]) continue;
                ++sanctions;
                int k = sanction_inducer[j];
                if (k >= 0) { ++induced_sanctions; ++out->induced_by[k]; }
                if (P.penalty_mode == 0) { ++fresh_bans; if (k >= 0) ++induced_bans; }
            }
            for (int j = 0; j < N; ++j) {
                if (capital[j] > 0.0 && next_capital[j] <= 0.0) {
                    ++bankruptcies;
                    if (timers[j] > 0 && ban_inducer[j] >= 0) {
                        ++induced_bankruptcies;
                        ++out->kills_by[ban_inducer[j]];
                    }
                }
            }
        }

        // Transition
        int active_next = 0;
        for (int i = 0; i < N; ++i) {
            capital[i] = next_capital[i];
            timers[i] = next_timers[i];
            if (timers[i] == 0) ban_inducer[i] = -1;
            lag_idx[i] = act_idx[i];
            active_next += (timers[i] == 0 && capital[i] > 0.0) ? 1 : 0;
        }
        ++episode_step;

        if (episode_step >= P.episode_length || active_next <= 1) {
            for (int i = 0; i < N; ++i)
                inactive[i] = (timers[i] > 0 || capital[i] <= 0.0) ? 1 : 0;
            for (int i = 0; i < N; ++i)
                if (P.policies[i] == POL_Q && pend[i].open)
                    close(i, pend[i], sim.encode(lag_idx.data(), inactive.data(), capital[i]), false);
            reset(ema_ref);
            episode_step = 0;
            if (step >= eval_start && step + 1 < P.total_steps) ++episodes_in_eval;
        }
    }

    // Truncate open transitions so repeated calls start clean
    for (int i = 0; i < N; ++i)
        inactive[i] = (timers[i] > 0 || capital[i] <= 0.0) ? 1 : 0;
    for (int i = 0; i < N; ++i)
        if (P.policies[i] == POL_Q && pend[i].open)
            close(i, pend[i], sim.encode(lag_idx.data(), inactive.data(), capital[i]), false);

    for (int i = 0; i < N; ++i) {
        out->mean_prices[i] = price_cnt[i] > 0 ? price_sum[i] / price_cnt[i] : P.nash_prices[i];
        out->mean_profits[i] = eval_periods > 0 ? profit_sum[i] / eval_periods : 0.0;
    }
    out->counts[0] = fresh_bans; out->counts[1] = induced_bans;
    out->counts[2] = sanctions; out->counts[3] = induced_sanctions;
    out->counts[4] = bankruptcies; out->counts[5] = induced_bankruptcies;
    out->counts[6] = episodes_in_eval; out->counts[7] = eval_periods;
    out->mean_cs[0] = eval_periods > 0 ? cs_sum / eval_periods : 0.0;
    return 0;
}
