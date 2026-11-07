#pragma once
#include <functional>
#include <vector>
namespace tokenomics {
struct CommitPlan {
    std::vector<int> tokens; // includes a terminal EOG token when sampled
    int accepted = 0;
    int prefix_after = 0;  // excludes the new pending (last emitted) token
    bool eog = false;
};
// Token acceptance is supplied by upstream. This only enforces stopping/counting.
CommitPlan plan_commit(int prefix_before, int remaining, int drafted,
                       const std::vector<int> & verified, const std::function<bool(int)> & is_eog);
int draft_budget(int requested, int remaining, int context_free);
}
