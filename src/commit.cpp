#include "commit.h"
#include <algorithm>
#include <stdexcept>
namespace tokenomics {
int draft_budget(int requested, int remaining, int context_free) {
    if (requested < 0 || remaining < 0 || context_free < 0) throw std::invalid_argument("negative budget");
    return std::max(0, std::min({requested, remaining - 1, context_free - 1}));
}
CommitPlan plan_commit(int prefix, int remaining, int drafted, const std::vector<int> & ids,
                       const std::function<bool(int)> & is_eog) {
    if (prefix < 0 || remaining <= 0 || drafted < 0 || ids.empty() || ids.size() > size_t(drafted + 1))
        throw std::invalid_argument("invalid verification result");
    CommitPlan result;
    for (int id : ids) {
        if (int(result.tokens.size()) == remaining) break;
        result.tokens.push_back(id);
        if (is_eog(id)) { result.eog = true; break; }
    }
    // The final verified token is correction/bonus; earlier ones match drafts.
    result.accepted = std::min(int(result.tokens.size()), int(ids.size()) - 1);
    result.prefix_after = prefix + int(result.tokens.size());
    return result;
}
}
