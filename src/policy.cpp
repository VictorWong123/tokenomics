#include "policy.h"
#include <algorithm>
#include <stdexcept>
namespace tokenomics {
bool valid_action(int k) { return k == 0 || std::find(caps.begin(), caps.end(), k) != caps.end(); }
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
    void reset() override { level_ = 1; streak_ = 0; }
    void observe(const Observation & o) override {
        if (o.drafted == 0) { streak_ = 0; return; }
        if (o.accepted == o.drafted) {
            if (++streak_ == 2) { level_ = std::min(level_ + 1, caps.size() - 1); streak_ = 0; }
        } else {
            streak_ = 0;
            if (2 * o.accepted < o.drafted && level_ > 0) --level_;
        }
    }
};
class Script final : public Policy {
    std::vector<int> actions_;
    size_t index_ = 0;
public:
    explicit Script(std::vector<int> a) : actions_(std::move(a)) {
        if (actions_.empty()) throw std::invalid_argument("empty diagnostic script");
        for (int k : actions_) if (!valid_action(k)) throw std::invalid_argument("invalid script action");
    }
    int select_action(const DecisionContext &) override { return actions_[index_++ % actions_.size()]; }
    void observe(const Observation &) override {}
    void reset() override { index_ = 0; }
};
std::unique_ptr<Policy> make_policy(const std::string & mode, int k, const std::vector<int> & script) {
    if (!valid_action(k) || k == 0) throw std::invalid_argument("k must be one of 1,2,4,6,8");
    if (mode == "target") return std::make_unique<Fixed>(0);
    if (mode == "fixed" || mode == "confidence") return std::make_unique<Fixed>(k);
    if (mode == "acceptance") return std::make_unique<Acceptance>();
    if (mode == "script") return std::make_unique<Script>(script);
    throw std::invalid_argument("unknown policy (cost-aware remains feasibility-gated)");
}
}
