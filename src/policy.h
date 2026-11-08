#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenomics {
inline const std::vector<int> caps{1, 2, 4, 6, 8};
inline constexpr std::array<int, 6> actions{0, 1, 2, 4, 6, 8};
bool valid_action(int k);
std::size_t action_index(int action);
struct DecisionContext {
    int cycle = 0, context_tokens = 0, remaining = 0, stale_tokens = 0;
};
struct Observation {
    int requested = 0, drafted = 0, accepted = 0, committed = 0;
    double wall_ms = 0;
    double sync_ms = 0;
    int synced_tokens = 0, effective_cap = -1;
};
struct ActionEstimate {
    double committed_tokens = 0, base_ms = 0;
};
struct ControllerConfig {
    std::string prior_id;
    bool synthetic = false;
    std::array<ActionEstimate, 6> priors{};
    double alpha = 0.2, switch_margin = 0.05, sync_ms_per_token = 0;
    int min_hold_cycles = 4, probe_interval = 16;
    bool allow_target_only = true, enable_probes = true;
    void validate() const;
};
struct ActionScore {
    int action = 0;
    bool eligible = false;
    double committed_tokens = 0, base_ms = 0, transition_ms = 0, score = 0;
};
enum class DecisionReason { initial, hold, margin, best, probe, capacity, budget_tail };
const char *decision_reason_name(DecisionReason reason);
struct DecisionTrace {
    std::array<ActionScore, 6> candidates{};
    int selected = 0, normal_action = 0;
    bool probe = false, normal_switch = false, forced_tail = false;
    bool learning_applied = false, tail_limited = false;
    DecisionReason reason = DecisionReason::initial;
    double predicted_tokens = 0, predicted_ms = 0, predicted_sync_ms = 0;
    double observed_base_ms = 0, estimate_tokens_after = 0, estimate_base_ms_after = 0;
    double sync_ms_per_token_after = 0;
};
class Policy {
  public:
    virtual ~Policy() = default;
    virtual int select_action(const DecisionContext &) = 0;
    virtual void observe(const Observation &) = 0;
    virtual void reset() = 0;
    virtual const DecisionTrace *trace() const { return nullptr; }
};
// Acceptance heuristic from the project proposal, not an established optimal policy.
std::unique_ptr<Policy> make_policy(const std::string &mode, int k,
                                    const std::vector<int> &script = {},
                                    const ControllerConfig *controller = nullptr);
} // namespace tokenomics
