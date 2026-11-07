#pragma once
#include <memory>
#include <string>
#include <vector>

namespace tokenomics {
inline const std::vector<int> caps{1, 2, 4, 6, 8};
bool valid_action(int k);
struct DecisionContext {
    int cycle = 0, context_tokens = 0, remaining = 0, stale_tokens = 0;
};
struct Observation {
    int requested = 0, drafted = 0, accepted = 0, committed = 0;
    double wall_ms = 0;
};
class Policy {
  public:
    virtual ~Policy() = default;
    virtual int select_action(const DecisionContext &) = 0;
    virtual void observe(const Observation &) = 0;
    virtual void reset() = 0;
};
// Acceptance heuristic from the project proposal, not an established optimal policy.
std::unique_ptr<Policy> make_policy(const std::string &mode, int k,
                                    const std::vector<int> &script = {});
} // namespace tokenomics
