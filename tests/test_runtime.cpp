#include "runtime.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace tokenomics;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error("check failed: " #x);                                         \
    } while (0)
static ControllerConfig controller(int variant = 0) {
    ControllerConfig c;
    c.prior_id = "synthetic-runtime-test-v1";
    c.synthetic = true;
    c.probe_interval = 2; // Exhaust transitions quickly; production default remains 16.
    c.allow_target_only = variant != 1;
    c.enable_probes = variant != 2;
    for (auto &p : c.priors)
        p = {1, 1e9};
    c.priors[0] = {1, 1e-9};
    return c;
}
static void check_totals(const Result &r) {
    CHECK(r.controller_active);
    CHECK(r.execution.cycles == r.cycles.size());
    std::uint64_t tokens = 0, drafted = 0, accepted = 0, synced = 0;
    for (const auto &c : r.cycles) {
        CHECK(c.controller.has_value());
        CHECK(c.action == c.controller->selected && c.effective_cap == c.action);
        CHECK(c.wall_ms >= c.observation_ms && c.policy_update_ms >= 0);
        CHECK(c.controller->learning_applied && !c.controller->tail_limited);
        tokens += c.committed;
        drafted += c.actual;
        accepted += c.accepted;
        synced += c.synced;
    }
    CHECK(tokens == r.tokens.size() && drafted == r.execution.drafted);
    CHECK(accepted == r.execution.accepted && synced == r.execution.synced_tokens);
    CHECK(r.execution.verified_rows == r.execution.cycles + drafted);
    CHECK(r.controller.learning_updates == r.execution.cycles);
    std::uint64_t by_action = 0;
    for (const auto &a : r.execution.actions)
        by_action += a.committed_tokens;
    CHECK(by_action == tokens);
}
int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    std::string dir = argv[1];
    int comparisons = 0;
    bool zero = false, partial = false, full = false, catchup = false, switched = false;
    bool controller_catchup = false, controller_fallback = false, controller_probe = false;
    std::array<bool, 6> controller_actions{};
    for (const std::string name :
         {"target", "random", "eos", "quantized", "quantized-k", "quantized-k-wide"}) {
        EngineConfig ec{dir + "/" + name + ".gguf", dir + "/" + name + ".gguf", 64, 32, 1, 0};
        Engine e(ec);
        for (int cap : {0, 1, 2, 7, 19, 80}) {
            Request base;
            base.prompt = "a b";
            base.max_tokens = cap;
            const auto expected = e.generate(base);
            for (auto cache : {"retain", "upstream"})
                for (auto drafter : {"model", "ngram"})
                    for (int variant = 0; variant < 3; ++variant) {
                        Request adaptive = base;
                        adaptive.mode = "cost-aware";
                        adaptive.cache = cache;
                        adaptive.drafter = drafter;
                        adaptive.controller = controller(variant);
                        if (variant == 1)
                            adaptive.proposal_transform = [](std::vector<int> &ids, int cycle) {
                                if (!ids.empty())
                                    ids[std::min(size_t(cycle % 3), ids.size() - 1)] = 260;
                            };
                        const auto result = e.generate(adaptive);
                        CHECK(result.tokens == expected.tokens && result.stop == expected.stop);
                        check_totals(result);
                        for (const auto &c : result.cycles) {
                            controller_actions[action_index(c.action)] = true;
                            controller_catchup |= c.synced > 1;
                            controller_probe |= c.controller->probe;
                            if (variant == 1)
                                CHECK(c.action != 0 || c.controller->forced_tail);
                            if (variant == 2)
                                CHECK(!c.controller->probe);
                        }
                        controller_fallback |= result.controller.fallback_cycles > 0;
                        adaptive.detailed = false;
                        const auto minimal = e.generate(adaptive);
                        CHECK(minimal.cycles.empty() && minimal.controller_active);
                        CHECK(minimal.tokens == result.tokens && minimal.stop == result.stop);
                        CHECK(minimal.execution.cycles > 0 || cap == 0);
                        ++comparisons;
                    }
            for (auto cache : {"retain", "upstream"})
                for (auto mode : {"fixed", "confidence", "acceptance", "script"}) {
                    Request r = base;
                    r.mode = mode;
                    r.cache = cache;
                    r.script = {0, 0, 4, 1, 0, 8, 2};
                    r.k = 8;
                    auto actual = e.generate(r);
                    CHECK(actual.tokens == expected.tokens);
                    CHECK(actual.stop == expected.stop);
                    ++comparisons;
                    for (auto c : actual.cycles) {
                        full |= c.actual > 0 && c.accepted == c.actual;
                        catchup |= c.synced > 1;
                        switched |= c.switched;
                    }
                    CHECK(e.generate(r).tokens == actual.tokens); // request reset
                }
            Request pert = base;
            pert.mode = "fixed";
            pert.k = 4;
            // Corrupt alternating proposal positions to force zero/partial acceptance.
            // Cache repair must never preserve a corrupted suffix.
            pert.proposal_transform = [](std::vector<int> &ids, int cycle) {
                if (!ids.empty())
                    ids[std::min(size_t(cycle % 3), ids.size() - 1)] = 260;
            };
            // Random fixtures exercise varying logit winners and partial commits.
            pert.diagnostic_logits = true;
            auto altered = e.generate(pert);
            for (const auto &cycle : altered.cycles) {
                CHECK(cycle.logits.size() == size_t(cycle.actual + 1));
                for (const auto &probe : cycle.logits) {
                    CHECK(probe.top_id != probe.second_id);
                    CHECK(probe.top_logit >= probe.second_logit);
                    if (probe.row < cycle.committed)
                        CHECK(altered.tokens.at(probe.output_index) == probe.top_id);
                }
            }
            CHECK(altered.tokens == expected.tokens);
            ++comparisons;
            for (auto c : altered.cycles) {
                zero |= c.actual > 0 && c.accepted == 0;
                partial |= c.accepted > 0 && c.accepted < c.actual;
            }
            Request ng = base;
            ng.mode = "fixed";
            ng.drafter = "ngram";
            ng.k = 8;
            CHECK(e.generate(ng).tokens == expected.tokens);
            ++comparisons;
        }
        Request boundary;
        boundary.prefix_tokens = std::vector<int>(64, 259);
        boundary.max_tokens = 10;
        auto b = e.generate(boundary);
        CHECK(b.tokens.size() == 1);
        boundary.mode = "fixed";
        boundary.k = 8;
        CHECK(e.generate(boundary).tokens == b.tokens);
        ++comparisons;
        boundary.mode = "cost-aware";
        boundary.controller = controller(1);
        auto tail = e.generate(boundary);
        CHECK(tail.tokens == b.tokens && tail.stop == b.stop);
        CHECK(tail.controller.forced_tail_cycles == 1);
        CHECK(tail.controller.fallback_cycles == 0);
        ++comparisons;
        boundary.prefix_tokens.push_back(259);
        bool throws = false;
        try {
            e.generate(boundary);
        } catch (const std::exception &) {
            throws = true;
        }
        CHECK(throws);
        // K-quant prefills exceed the tiled threshold, while target verification
        // remains on consistent dot kernels at every supported proposal cap.
        Request long_prefix;
        long_prefix.prefix_tokens = std::vector<int>(32, 259);
        long_prefix.max_tokens = 19;
        auto prefetched = e.generate(long_prefix);
        for (int k : {1, 2, 4, 6, 8}) {
            long_prefix.mode = "fixed";
            long_prefix.k = k;
            CHECK(e.generate(long_prefix).tokens == prefetched.tokens);
            ++comparisons;
        }
        auto probe_request = long_prefix;
        probe_request.mode = "target";
        probe_request.max_cycles = 1;
        probe_request.diagnostic_logits = true;
        auto reference_probe = e.generate(probe_request).cycles.at(0).logits.at(0);
        for (int k : {1, 2, 4, 6, 8}) {
            probe_request.mode = "fixed";
            probe_request.k = k;
            auto detailed = e.generate(probe_request);
            auto probe = detailed.cycles.at(0).logits.at(0);
            CHECK(probe.top_id == reference_probe.top_id);
            CHECK(probe.second_id == reference_probe.second_id);
            CHECK(probe.top_logit == reference_probe.top_logit);
            CHECK(probe.second_logit == reference_probe.second_logit);
            auto minimal_request = probe_request;
            minimal_request.detailed = false;
            minimal_request.diagnostic_logits = false;
            auto minimal = e.generate(minimal_request);
            CHECK(minimal.cycles.empty());
            CHECK(minimal.tokens == detailed.tokens && minimal.stop == detailed.stop);
            CHECK(minimal.final_target_cached == detailed.final_target_cached);
            CHECK(minimal.final_draft_cached == detailed.final_draft_cached);
        }
        Request empty;
        empty.prompt = "";
        empty.max_tokens = 2;
        CHECK(!e.generate(empty).prompt_ids.empty());
        Request capture;
        capture.prompt = "a b";
        capture.prefix_tokens = std::vector<int>(16, 259);
        capture.max_tokens = 32;
        capture.capture_checkpoints = {1, 12};
        capture.checkpoint_namespace = "trace";
        capture.diagnostic_logits = true;
        const auto trace = e.generate(capture);
        CHECK((trace.checkpoint_bytes > 0) == !trace.checkpoint_ids.empty());
        for (int cp : {1, 12}) {
            const std::string id = "trace:" + std::to_string(cp);
            if (std::find(trace.checkpoint_ids.begin(), trace.checkpoint_ids.end(), id) ==
                trace.checkpoint_ids.end())
                continue; // EOS checkpoints cannot be continued.
            Request mismatched_prefix;
            mismatched_prefix.resume_checkpoint = id;
            mismatched_prefix.prefix_tokens = trace.prompt_ids;
            bool mismatch_throws = false;
            try {
                e.generate(mismatched_prefix);
            } catch (const std::exception &) {
                mismatch_throws = true;
            }
            CHECK(mismatch_throws);
            for (int action : {0, 1, 2, 4, 6, 8}) {
                Request replay;
                replay.resume_checkpoint = id;
                replay.mode = action == 0 ? "target" : "fixed";
                replay.k = action == 0 ? 4 : action;
                replay.max_tokens = 9;
                replay.diagnostic_logits = true;
                auto restored = e.generate(replay);
                const auto end = std::min(trace.tokens.size(), size_t(cp + 9));
                CHECK(restored.tokens ==
                      std::vector<int>(trace.tokens.begin() + cp, trace.tokens.begin() + end));
                CHECK(restored.prompt_ids.size() == trace.prompt_ids.size() + cp);
                const auto &original = trace.cycles.at(cp).logits.at(0);
                const auto &resumed = restored.cycles.at(0).logits.at(0);
                CHECK(original.top_id == resumed.top_id);
                CHECK(original.top_logit == resumed.top_logit);
                CHECK(original.second_logit == resumed.second_logit);
                CHECK(original.second_id == resumed.second_id);
                CHECK(restored.cycles.at(0).synced == 0);
                ++comparisons;
            }
        }
        (void)e.generate(empty); // A new ordinary request releases diagnostic storage.
        Request expired;
        expired.resume_checkpoint = "trace:1";
        bool expired_throws = false;
        try {
            e.generate(expired);
        } catch (const std::exception &) {
            expired_throws = true;
        }
        CHECK(expired_throws);
    }
    Engine wrong({dir + "/target.gguf", dir + "/wrong.gguf", 64, 32, 1, 0});
    Request r;
    r.mode = "fixed";
    r.prompt = "a";
    r.max_tokens = 19;
    r.k = 8;
    auto w = wrong.generate(r);
    CHECK(w.tokens == std::vector<int>(19, 259));
    Request replay;
    replay.prefix_tokens = std::vector<int>(10, 259);
    replay.max_tokens = 1;
    auto state = wrong.generate(replay);
    CHECK(state.final_draft_cached == 9 && state.final_target_cached == 10);
    CHECK(state.tokens == std::vector<int>{259});
    r.diagnostic_logits = true;
    auto diagnostic = wrong.generate(r);
    CHECK(diagnostic.tokens == w.tokens);
    for (const auto &cycle : diagnostic.cycles) {
        CHECK(cycle.logits.size() == size_t(cycle.actual + 1));
        for (const auto &probe : cycle.logits) {
            CHECK(probe.top_id == 259);
            CHECK(probe.top_logit >= probe.second_logit);
            if (probe.row < cycle.committed)
                CHECK(diagnostic.tokens.at(probe.output_index) == probe.top_id);
        }
    }
    CHECK(zero && partial && full && catchup && switched);
    CHECK(controller_catchup && controller_fallback && controller_probe);
    CHECK(std::all_of(controller_actions.begin(), controller_actions.end(),
                      [](bool seen) { return seen; }));
    // A period-nine history makes upstream ngram-simple propose up to eight
    // tokens. A constant-token fixture can return no draft because its nearest
    // match has too little following history, so it cannot cover this path.
    Engine lookup({dir + "/ngram-cycle.gguf", "", 64, 32, 1, 0});
    Request lookup_base;
    lookup_base.prompt = "a b";
    lookup_base.max_tokens = 40;
    const auto lookup_reference = lookup.generate(lookup_base);
    bool lookup_zero = false, lookup_partial = false, lookup_switch = false;
    for (int action : {1, 2, 4, 6, 8, -1}) {
        Request candidate = lookup_base;
        candidate.drafter = "ngram";
        candidate.mode = action < 0 ? "script" : "fixed";
        candidate.k = action < 0 ? 4 : action;
        candidate.script = {8, 0, 0, 2, 0, 6};
        const auto correct = lookup.generate(candidate);
        CHECK(correct.tokens == lookup_reference.tokens && correct.stop == lookup_reference.stop);
        CHECK(lookup.generate(candidate).tokens == correct.tokens);
        bool proposed = false, reached_cap = false;
        for (const auto &cycle : correct.cycles) {
            CHECK(cycle.actual <= cycle.action);
            CHECK(cycle.draft_cached == 0 && cycle.synced == 0);
            CHECK(cycle.accepted == cycle.actual);
            proposed |= cycle.actual > 0;
            reached_cap |= cycle.actual == action;
            lookup_switch |= cycle.switched;
        }
        CHECK(proposed);
        CHECK(action < 0 || reached_cap);
        ++comparisons;
        candidate.proposal_transform = [](std::vector<int> &ids, int cycle) {
            if (!ids.empty())
                ids[std::min(size_t(cycle % 3), ids.size() - 1)] = 260;
        };
        const auto corrupt = lookup.generate(candidate);
        CHECK(corrupt.tokens == lookup_reference.tokens && corrupt.stop == lookup_reference.stop);
        for (const auto &cycle : corrupt.cycles) {
            lookup_zero |= cycle.actual > 0 && cycle.accepted == 0;
            lookup_partial |= cycle.accepted > 0 && cycle.accepted < cycle.actual;
        }
        ++comparisons;
    }
    CHECK(lookup_zero && lookup_partial && lookup_switch);
    std::cout << comparisons
              << " differential fixture cases passed; zero/partial/full acceptance, EOS, rollback, "
                 "switching, catch-up, reset, budgets and context exercised\n";
}
