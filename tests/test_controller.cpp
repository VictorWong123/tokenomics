#include "policy.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
using namespace tokenomics;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error("check failed: " #x);                                         \
    } while (0)
static bool near(double a, double b) { return std::abs(a - b) < 1e-10; }
template <class F> static void invalid(F f) {
    bool threw = false;
    try {
        f();
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}
static ControllerConfig prior() {
    ControllerConfig c;
    c.prior_id = "synthetic-policy-test-v1";
    c.synthetic = true;
    for (auto &p : c.priors)
        p = {1, 100};
    c.priors[0] = {1, 1};
    return c;
}
static Observation outcome(int action, double wall, int committed = 1) {
    return {action, committed - 1, committed - 1, committed, wall, 0, 0, action};
}
static DecisionContext context(int cycle = 0, int remaining = 100, int stale = 0) {
    return {cycle, 100, remaining, stale};
}
int main() {
    auto cfg = prior();
    cfg.priors[0] = {1, 10};
    cfg.priors[1] = {2, 2};
    cfg.priors[2] = {1, 2};
    cfg.enable_probes = false;
    auto p = make_policy("cost-aware", 4, {}, &cfg);
    CHECK(p->select_action(context()) == 1);
    CHECK(near(p->trace()->predicted_tokens, 2));
    p->observe(outcome(1, 4));
    CHECK(near(p->trace()->estimate_tokens_after, 1.8));
    CHECK(near(p->trace()->estimate_base_ms_after, 2.4));
    CHECK(p->trace()->learning_applied);
    p->select_action(context(1));
    CHECK(near(p->trace()->candidates[0].base_ms, 10));
    CHECK(near(p->trace()->candidates[2].base_ms, 2));
    p->reset();
    CHECK(p->select_action(context()) == 1);
    CHECK(near(p->trace()->predicted_ms, 2));

    cfg = prior();
    cfg.alpha = 1;
    cfg.enable_probes = false;
    cfg.priors[1] = {1, 1.04};
    p = make_policy("cost-aware", 4, {}, &cfg);
    for (int i = 0; i < 4; ++i) {
        CHECK(p->select_action(context(i)) == 0);
        if (i)
            CHECK(p->trace()->reason == DecisionReason::hold);
        p->observe(outcome(0, 2));
    }
    CHECK(p->select_action(context(4)) == 1);
    CHECK(p->trace()->normal_switch && p->trace()->reason == DecisionReason::best);

    cfg = prior();
    cfg.alpha = 1;
    cfg.enable_probes = false;
    cfg.min_hold_cycles = 1;
    cfg.priors[1] = {1, 1};
    p = make_policy("cost-aware", 4, {}, &cfg);
    CHECK(p->select_action(context()) == 0); // exact tie favors ordinary target
    p->observe(outcome(0, 1.04));
    CHECK(p->select_action(context(1)) == 0); // only 4% better, below switch margin
    CHECK(p->trace()->reason == DecisionReason::margin);
    p->observe(outcome(0, 1.06));
    CHECK(p->select_action(context(2)) == 1);

    cfg = prior();
    p = make_policy("cost-aware", 4, {}, &cfg);
    std::vector<int> probes;
    for (int i = 0; i <= 80; ++i) {
        const int a = p->select_action(context(i));
        CHECK(p->trace()->normal_action == 0);
        if (i > 0 && i % 16 == 0) {
            CHECK(p->trace()->probe && a != 0);
            probes.push_back(a);
        } else
            CHECK(!p->trace()->probe && a == 0);
        p->observe(outcome(a, a == 0 ? 1 : 100));
    }
    CHECK(probes == std::vector<int>({1, 2, 4, 6, 8}));
    p->reset();
    CHECK(p->select_action(context()) == 0);

    cfg = prior();
    cfg.alpha = 1;
    cfg.min_hold_cycles = 1;
    p = make_policy("cost-aware", 4, {}, &cfg);
    for (int i = 0; i < 16; ++i) {
        CHECK(p->select_action(context(i)) == 0);
        p->observe(outcome(0, 1));
    }
    CHECK(p->select_action(context(16)) == 1 && p->trace()->probe);
    p->observe(outcome(1, 0.1, 2));
    CHECK(p->select_action(context(17)) == 1 && !p->trace()->probe);

    cfg = prior();
    cfg.priors[1] = {2, 1};
    cfg.sync_ms_per_token = 1;
    cfg.probe_interval = 1;
    p = make_policy("cost-aware", 4, {}, &cfg);
    CHECK(p->select_action(context(0, 100, 10)) == 0);
    p->observe(outcome(0, 1));
    CHECK(p->select_action(context(1, 100, 10)) == 1 && p->trace()->probe);
    CHECK(near(p->trace()->predicted_sync_ms, 10));
    CHECK(near(p->trace()->predicted_ms, 11));
    p->observe({1, 1, 1, 2, 22, 20, 10, 1});
    CHECK(near(p->trace()->observed_base_ms, 2));
    CHECK(near(p->trace()->estimate_base_ms_after, 1.2));
    CHECK(near(p->trace()->sync_ms_per_token_after, 1.2));
    p->select_action(context(2, 100, 0));
    CHECK(near(p->trace()->candidates[1].base_ms, 1.2));
    CHECK(near(p->trace()->candidates[1].transition_ms, 0));

    cfg = prior();
    cfg.enable_probes = false;
    p = make_policy("cost-aware", 4, {}, &cfg);
    for (int i = 0; i < 100; ++i) {
        CHECK(p->select_action(context(i)) == 0 && !p->trace()->probe);
        p->observe(outcome(0, 1));
    }
    cfg.allow_target_only = false;
    p = make_policy("cost-aware", 4, {}, &cfg);
    CHECK(p->select_action(context(0, 2)) == 1);
    CHECK(!p->trace()->candidates[0].eligible);
    p->observe(outcome(1, 1));
    CHECK(p->select_action(context(1, 1)) == 0 && p->trace()->forced_tail);
    CHECK(p->trace()->reason == DecisionReason::budget_tail);

    cfg = prior();
    cfg.priors[5] = {9, 1};
    p = make_policy("cost-aware", 4, {}, &cfg);
    CHECK(p->select_action(context()) == 8);
    p->observe({8, 1, 1, 2, 1, 0, 0, 1});
    CHECK(p->trace()->tail_limited && !p->trace()->learning_applied);
    CHECK(near(p->trace()->estimate_tokens_after, 9));
    CHECK(near(p->trace()->estimate_base_ms_after, 1));
    p->select_action(context(1, 3));
    CHECK(!p->trace()->candidates[5].eligible);

    invalid([] { make_policy("cost-aware", 4); });
    for (int kind = 0; kind < 7; ++kind) {
        cfg = prior();
        if (kind == 0)
            cfg.prior_id.clear();
        if (kind == 1)
            cfg.alpha = 0;
        if (kind == 2)
            cfg.switch_margin = -1;
        if (kind == 3)
            cfg.probe_interval = 0;
        if (kind == 4)
            cfg.priors[0].base_ms = 1e-320;
        if (kind == 5)
            cfg.priors[1].committed_tokens = 3;
        if (kind == 6)
            cfg.sync_ms_per_token = std::numeric_limits<double>::infinity();
        invalid([&] { cfg.validate(); });
    }
    cfg = prior();
    p = make_policy("cost-aware", 4, {}, &cfg);
    invalid([&] { p->observe(outcome(0, 1)); });
    invalid([&] { p->select_action(context(0, 0)); });
    CHECK(p->select_action(context()) == 0);
    invalid([&] { p->select_action(context()); });
    invalid([&] { p->observe(outcome(1, 1)); });
    invalid([&] { p->observe(outcome(0, std::numeric_limits<double>::quiet_NaN())); });
    p->observe(outcome(0, 1)); // invalid samples left estimates/pending state untouched
    invalid([&] { p->observe(outcome(0, 1)); });

    std::vector<int> first;
    cfg = prior();
    cfg.probe_interval = 7;
    p = make_policy("cost-aware", 4, {}, &cfg);
    for (int run = 0; run < 2; ++run) {
        p->reset();
        std::mt19937 rng(20261108);
        for (int i = 0; i < 500; ++i) {
            const int remaining = 1 + int(rng() % 24), stale = int(rng() % 200);
            const int a = p->select_action(context(i, remaining, stale));
            CHECK(valid_action(a) && (a == 0 || a < remaining));
            if (!run)
                first.push_back(a);
            else
                CHECK(first.at(i) == a);
            const int drafted = int(rng() % unsigned(a + 1));
            const int accepted = int(rng() % unsigned(drafted + 1));
            const double wall = 1 + double(rng() % 10000) / 100;
            const int synced = a > 0 ? stale : 0;
            p->observe(
                {a, drafted, accepted, accepted + 1, wall, synced > 0 ? wall / 3 : 0, synced, a});
        }
    }
    std::cout << "controller arithmetic, stability, recovery, reset and boundary checks passed\n";
}
