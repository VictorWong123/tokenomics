#include "runtime.h"
#include <iostream>
#include <stdexcept>
using namespace tokenomics;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error("check failed: " #x);                                         \
    } while (0)
int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    std::string dir = argv[1];
    int comparisons = 0;
    bool zero = false, partial = false, full = false, catchup = false, switched = false;
    for (const std::string name : {"target", "random", "eos", "quantized", "quantized-k"}) {
        EngineConfig ec{dir + "/" + name + ".gguf", dir + "/" + name + ".gguf", 64, 32, 1, 0};
        Engine e(ec);
        for (int cap : {0, 1, 2, 7, 19, 80}) {
            Request base;
            base.prompt = "a b";
            base.max_tokens = cap;
            const auto expected = e.generate(base);
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
    std::cout << comparisons
              << " differential fixture cases passed; zero/partial/full acceptance, EOS, rollback, "
                 "switching, catch-up, reset, budgets and context exercised\n";
}
