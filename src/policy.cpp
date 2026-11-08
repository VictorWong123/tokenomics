#include "policy.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace tokenomics {
bool valid_action(int k) { return k == 0 || std::find(caps.begin(), caps.end(), k) != caps.end(); }
std::size_t action_index(int action) {
    const auto it = std::find(actions.begin(), actions.end(), action);
    if (it == actions.end())
        throw std::invalid_argument("unsupported action");
    return std::size_t(it - actions.begin());
}
void ControllerConfig::validate() const {
    if (prior_id.empty() || prior_id.size() > 256 || !std::isfinite(alpha) || alpha <= 0 ||
        alpha > 1 || !std::isfinite(switch_margin) || switch_margin < 0 || switch_margin > 1 ||
        !std::isfinite(sync_ms_per_token) || sync_ms_per_token < 0 || sync_ms_per_token > 1e9 ||
        min_hold_cycles < 1 || min_hold_cycles > 1000000 || probe_interval < 1 ||
        probe_interval > 1000000)
        throw std::invalid_argument("invalid controller parameters or missing frozen prior ID");
    for (std::size_t i = 0; i < priors.size(); ++i) {
        const auto &p = priors[i];
        if (!std::isfinite(p.committed_tokens) || p.committed_tokens < 1 ||
            p.committed_tokens > actions[i] + 1 || !std::isfinite(p.base_ms) || p.base_ms < 1e-9 ||
            p.base_ms > 1e12)
            throw std::invalid_argument(
                "all six controller priors need feasible tokens and positive costs");
    }
}
const char *decision_reason_name(DecisionReason reason) {
    switch (reason) {
    case DecisionReason::initial:
        return "initial";
    case DecisionReason::hold:
        return "hold";
    case DecisionReason::margin:
        return "margin";
    case DecisionReason::best:
        return "best";
    case DecisionReason::probe:
        return "probe";
    case DecisionReason::capacity:
        return "capacity";
    case DecisionReason::budget_tail:
        return "budget_tail";
    }
    throw std::invalid_argument("invalid decision reason");
}
class CostAware final : public Policy {
    ControllerConfig cfg_;
    std::array<ActionEstimate, 6> estimates_{};
    double sync_per_token_ = 0;
    int normal_ = -1;
    std::uint64_t completed_ = 0, normal_age_ = 0;
    std::size_t probe_cursor_ = 0;
    bool pending_ = false;
    DecisionTrace trace_;

  public:
    explicit CostAware(const ControllerConfig &cfg) : cfg_(cfg) {
        cfg_.validate();
        reset();
    }
    void reset() override {
        estimates_ = cfg_.priors;
        sync_per_token_ = cfg_.sync_ms_per_token;
        normal_ = -1;
        completed_ = normal_age_ = 0;
        probe_cursor_ = 0;
        pending_ = false;
        trace_ = {};
    }
    int select_action(const DecisionContext &ctx) override {
        if (pending_ || ctx.cycle < 0 || ctx.context_tokens < 0 || ctx.remaining < 1 ||
            ctx.stale_tokens < 0)
            throw std::invalid_argument("invalid controller context or unobserved decision");
        trace_ = {};
        int best = -1;
        for (std::size_t i = 0; i < actions.size(); ++i) {
            auto &s = trace_.candidates[i];
            s.action = actions[i];
            s.eligible = s.action == 0 ? cfg_.allow_target_only || ctx.remaining == 1
                                       : s.action <= ctx.remaining - 1;
            s.committed_tokens = estimates_[i].committed_tokens;
            s.base_ms = estimates_[i].base_ms;
            s.transition_ms = s.action == 0 ? 0 : ctx.stale_tokens * sync_per_token_;
            s.score = s.eligible ? s.committed_tokens / (s.base_ms + s.transition_ms) : 0;
            // Action order supplies the conservative lower-action tie break.
            if (s.eligible && (best < 0 || s.score > trace_.candidates[best].score))
                best = int(i);
        }
        if (best < 0)
            throw std::logic_error("controller has no eligible action");
        int chosen = best;
        if (ctx.remaining == 1) {
            trace_.forced_tail = true;
            trace_.reason = DecisionReason::budget_tail;
        } else {
            if (normal_ < 0 || !trace_.candidates[normal_].eligible) {
                trace_.reason = normal_ < 0 ? DecisionReason::initial : DecisionReason::capacity;
                trace_.normal_switch = normal_ >= 0 && normal_ != best;
                normal_ = best;
                normal_age_ = 0;
            } else if (normal_age_ < std::uint64_t(cfg_.min_hold_cycles)) {
                trace_.reason = DecisionReason::hold;
            } else if (best != normal_ &&
                       trace_.candidates[best].score >=
                           trace_.candidates[normal_].score * (1 + cfg_.switch_margin)) {
                normal_ = best;
                normal_age_ = 0;
                trace_.normal_switch = true;
                trace_.reason = DecisionReason::best;
            } else {
                trace_.reason = DecisionReason::margin;
            }
            chosen = normal_;
            if (cfg_.enable_probes && completed_ > 0 &&
                completed_ % std::uint64_t(cfg_.probe_interval) == 0) {
                for (std::size_t offset = 0; offset < actions.size(); ++offset) {
                    const auto candidate = (probe_cursor_ + offset) % actions.size();
                    if (int(candidate) != normal_ && trace_.candidates[candidate].eligible) {
                        chosen = int(candidate);
                        probe_cursor_ = (candidate + 1) % actions.size();
                        trace_.probe = true;
                        trace_.reason = DecisionReason::probe;
                        break;
                    }
                }
            }
        }
        const auto &s = trace_.candidates[chosen];
        trace_.selected = s.action;
        trace_.normal_action = normal_ < 0 ? s.action : actions[normal_];
        trace_.predicted_tokens = s.committed_tokens;
        trace_.predicted_ms = s.base_ms + s.transition_ms;
        trace_.predicted_sync_ms = s.transition_ms;
        pending_ = true;
        return trace_.selected;
    }
    void observe(const Observation &o) override {
        if (!pending_ || o.requested != trace_.selected || !std::isfinite(o.wall_ms) ||
            o.wall_ms < 1e-9 || o.wall_ms > 1e12 || !std::isfinite(o.sync_ms) || o.sync_ms < 0 ||
            o.sync_ms >= o.wall_ms || o.synced_tokens < 0 || o.effective_cap < 0 ||
            o.effective_cap > o.requested || o.drafted < 0 || o.drafted > o.effective_cap ||
            o.accepted < 0 || o.accepted > o.drafted || o.committed < 1 ||
            o.committed < o.accepted || o.committed > o.accepted + 1 ||
            (o.requested == 0 && (o.synced_tokens != 0 || o.sync_ms != 0)))
            throw std::invalid_argument("invalid controller observation");
        const auto index = action_index(o.requested);
        const auto blend = [this](double prior, double sample) {
            return (1 - cfg_.alpha) * prior + cfg_.alpha * sample;
        };
        trace_.observed_base_ms = o.wall_ms - (o.synced_tokens > 0 ? o.sync_ms : 0);
        trace_.tail_limited = o.requested > 0 && o.effective_cap != o.requested;
        if (!trace_.tail_limited) {
            estimates_[index].committed_tokens =
                blend(estimates_[index].committed_tokens, o.committed);
            estimates_[index].base_ms = blend(estimates_[index].base_ms, trace_.observed_base_ms);
            trace_.learning_applied = true;
        }
        if (o.synced_tokens > 0)
            sync_per_token_ = blend(sync_per_token_, o.sync_ms / o.synced_tokens);
        trace_.estimate_tokens_after = estimates_[index].committed_tokens;
        trace_.estimate_base_ms_after = estimates_[index].base_ms;
        trace_.sync_ms_per_token_after = sync_per_token_;
        ++completed_;
        if (!trace_.probe && !trace_.forced_tail)
            ++normal_age_;
        pending_ = false;
    }
    const DecisionTrace *trace() const override { return &trace_; }
};
class Fixed final : public Policy {
    int k_;

  public:
    explicit Fixed(int k) : k_(k) {}
    int select_action(const DecisionContext &) override { return k_; }
    void observe(const Observation &) override {}
    void reset() override {}
};
class Acceptance final : public Policy {
    size_t level_ = 1;
    int streak_ = 0;

  public:
    int select_action(const DecisionContext &) override { return caps[level_]; }
    void reset() override {
        level_ = 1;
        streak_ = 0;
    }
    void observe(const Observation &o) override {
        if (o.drafted == 0) {
            streak_ = 0;
            return;
        }
        if (o.accepted == o.drafted) {
            if (++streak_ == 2) {
                level_ = std::min(level_ + 1, caps.size() - 1);
                streak_ = 0;
            }
        } else {
            streak_ = 0;
            if (2 * o.accepted < o.drafted && level_ > 0)
                --level_;
        }
    }
};
class Script final : public Policy {
    std::vector<int> actions_;
    size_t index_ = 0;

  public:
    explicit Script(std::vector<int> a) : actions_(std::move(a)) {
        if (actions_.empty())
            throw std::invalid_argument("empty diagnostic script");
        for (int k : actions_)
            if (!valid_action(k))
                throw std::invalid_argument("invalid script action");
    }
    int select_action(const DecisionContext &) override {
        return actions_[index_++ % actions_.size()];
    }
    void observe(const Observation &) override {}
    void reset() override { index_ = 0; }
};
std::unique_ptr<Policy> make_policy(const std::string &mode, int k, const std::vector<int> &script,
                                    const ControllerConfig *controller) {
    if (!valid_action(k) || k == 0)
        throw std::invalid_argument("k must be one of 1,2,4,6,8");
    if (mode == "target")
        return std::make_unique<Fixed>(0);
    if (mode == "fixed" || mode == "confidence")
        return std::make_unique<Fixed>(k);
    if (mode == "acceptance")
        return std::make_unique<Acceptance>();
    if (mode == "script")
        return std::make_unique<Script>(script);
    if (mode == "cost-aware") {
        if (!controller)
            throw std::invalid_argument(
                "cost-aware requires an explicit frozen controller configuration");
        return std::make_unique<CostAware>(*controller);
    }
    throw std::invalid_argument("unknown policy");
}
} // namespace tokenomics
