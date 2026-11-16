#include "logit_copy.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
using namespace tokenomics;
using json = nlohmann::json;
static void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
static std::vector<int> reference(const std::vector<int> &history, int pending, int cap,
                                  const std::vector<float> &logits, int minimum, int cutoff) {
    std::vector<int> ranked(logits.size());
    std::iota(ranked.begin(), ranked.end(), 0);
    std::partial_sort(ranked.begin(), ranked.begin() + cutoff, ranked.end(), [&](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    std::vector<int> ranks(logits.size(), -1);
    for (int i = 0; i < cutoff; ++i)
        if (ranked[i] != pending)
            ranks[ranked[i]] = i;
    auto observed = history;
    observed.push_back(pending);
    std::tuple<int, int, int> best{0, 0, 0};
    int chosen = -1;
    for (int length = minimum; length <= std::min(32, int(observed.size()) - 1); ++length)
        for (int start = 0; start + length < int(observed.size()); ++start)
            if (std::equal(observed.begin() + start, observed.begin() + start + length,
                           observed.end() - length)) {
                const int boundary = start + length;
                if (ranks[observed[boundary]] < 0)
                    continue;
                const auto score = std::tuple{length, -ranks[observed[boundary]], boundary};
                if (chosen < 0 || score > best) {
                    best = score;
                    chosen = boundary;
                }
            }
    return chosen < 0
               ? std::vector<int>{}
               : std::vector<int>(observed.begin() + chosen,
                                  observed.begin() + std::min(int(observed.size()), chosen + cap));
}
static void semantics() {
    int cases = 0;
    for (int size = 1, limit = 3; size <= 7; ++size, limit *= 3)
        for (int code = 0; code < limit; ++code) {
            std::vector<int> observed(size);
            int value = code;
            for (auto &id : observed) {
                id = value % 3;
                value /= 3;
            }
            const int pending = observed.back();
            observed.pop_back();
            std::vector<float> logits(67, -20.0f);
            logits[pending] = 50;
            logits[(pending + 1) % 3] = 20;
            logits[(pending + 2) % 3] = 20; // Ranking ties use the lower token ID.
            for (int minimum : {1, 3, 5})
                for (int cutoff : {2, 3})
                    for (int cap : {0, 1, 4}) {
                        std::vector<int> actual;
                        LogitCopyStats stats;
                        logit_copy_draft(observed, pending, cap, logits, actual, stats, minimum,
                                         cutoff);
                        check(actual == reference(observed, pending, cap, logits, minimum, cutoff),
                              "native guided proposal differs from independent sorted reference");
                        ++cases;
                    }
        }
    std::vector<int> output;
    LogitCopyStats stats;
    logit_copy_draft({}, 0, 4, {}, output, stats);
    check(output.empty() && stats.rank_checks == 0, "first cycle used a future hint");
    const std::vector<int> history{0, 1, 2, 0, 1};
    std::vector<float> nonfinite(67, 0.0f);
    nonfinite[2] = NAN;
    logit_copy_draft(history, 2, 4, nonfinite, output, stats, 1, 64);
    check(output.empty(), "nonfinite ranking produced a proposal");
    for (int candidate : {3, 63, 64, 65, 262})
        for (int kind = 0; kind < 4; ++kind) {
            const std::vector<int> prefix{0, 1, 2, candidate, 0, 1};
            std::vector<float> logits(263, kind == 0 ? 0.0f : -std::numeric_limits<float>::max());
            logits[2] = std::numeric_limits<float>::max();
            logits[candidate] = kind == 1   ? std::numeric_limits<float>::denorm_min()
                                : kind == 2 ? -0.0f
                                            : 0.0f;
            for (int cutoff : {2, 8, 64}) {
                LogitCopyStats check_stats;
                logit_copy_draft(prefix, 2, 4, logits, output, check_stats, 3, cutoff);
                check(output == reference(prefix, 2, 4, logits, 3, cutoff),
                      "rank boundary, signed-zero or subnormal membership differs");
                ++cases;
            }
        }
    std::vector<int> repeated;
    for (int i = 0; i < 40; ++i)
        repeated.push_back(i % 4);
    repeated.push_back(0);
    repeated.push_back(1);
    std::vector<float> just_outside(263, 0.0f);
    just_outside[2] = 100;
    just_outside[3] = 1;
    for (int i = 200; i < 263; ++i)
        just_outside[i] = 2;
    logit_copy_draft(repeated, 2, 8, just_outside, output, stats);
    check(output.empty() && stats.rank_checks == 1, "one source ID was repeatedly rescanned");
    std::cout << cases << " native sorted-reference proposal cases passed\n";
}
static void replay_file(const char *path) {
    std::ifstream stream(path);
    check(bool(stream), "cannot open canonical rank dataset");
    for (std::string line; std::getline(stream, line);) {
        const auto target = json::parse(line);
        const auto prompt = target.at("prompt_ids").get<std::vector<int>>();
        const auto tokens = target.at("token_ids").get<std::vector<int>>();
        for (int minimum : {1, 3, 5})
            for (int cutoff : {8, 64})
                for (int cap : {1, 4, 8}) {
                    std::vector<int> history(prompt.begin(), prompt.end() - 1), proposed;
                    int pending = prompt.back(), cursor = 0, cycles = 0, drafted = 0, accepted = 0;
                    std::vector<float> logits(151936, 0.0f);
                    std::uint64_t rank_checks = 0, examined = 0;
                    json trace = json::array();
                    while (cursor < int(tokens.size())) {
                        std::span<const float> prior;
                        if (cursor) {
                            std::fill(logits.begin(), logits.end(), 0.0f);
                            const auto ids = target.at("rows")
                                                 .at(cursor - 1)
                                                 .at("top_ids")
                                                 .get<std::vector<int>>();
                            check(ids.size() == 64 && ids[0] == pending,
                                  "wrong prior canonical row");
                            for (int rank = 0; rank < 64; ++rank) {
                                check(ids[rank] >= 0 && ids[rank] < int(logits.size()),
                                      "rank token outside pinned vocabulary");
                                logits[ids[rank]] = float(64 - rank);
                            }
                            prior = logits;
                        }
                        LogitCopyStats stats;
                        logit_copy_draft(history, pending,
                                         std::min(cap, int(tokens.size()) - cursor - 1), prior,
                                         proposed, stats, minimum, cutoff);
                        rank_checks += stats.rank_checks;
                        examined += stats.logits_examined;
                        trace.push_back(
                            {{"output_index", cursor},
                             {"proposal_ids", proposed},
                             {"hint_row_used", cursor ? json(cursor - 1) : json(nullptr)}});
                        int matches = 0;
                        while (matches < int(proposed.size()) &&
                               proposed[matches] == tokens[cursor + matches])
                            ++matches;
                        for (int i = 0; i <= matches; ++i) {
                            history.push_back(pending);
                            pending = tokens[cursor + i];
                        }
                        cursor += matches + 1;
                        ++cycles;
                        drafted += int(proposed.size());
                        accepted += matches;
                    }
                    std::cout << json({{"prompt_id", target.at("prompt_id")},
                                       {"minimum", minimum},
                                       {"cutoff", cutoff},
                                       {"cap", cap},
                                       {"cycles", cycles},
                                       {"drafted", drafted},
                                       {"accepted", accepted},
                                       {"verified_rows", cycles + drafted},
                                       {"wasted_rows", cycles + drafted - int(tokens.size())},
                                       {"rank_checks", rank_checks},
                                       {"logits_examined", examined},
                                       {"trace", trace},
                                       {"synthetic_rank_embedding", true},
                                       {"performance_claim", nullptr}})
                                     .dump()
                              << '\n';
                }
    }
}
static void benchmark() {
    using Clock = std::chrono::steady_clock;
    for (int size : {32768, 151936})
        for (int history_size : {128, 2048})
            for (int kind = 0; kind < 3; ++kind) {
                std::vector<int> history;
                for (int i = 0; i < history_size; ++i)
                    history.push_back(i % 4);
                history.push_back(0);
                history.push_back(1);
                const int pending = 2;
                std::mt19937 rng(20261116 + size + kind);
                std::vector<float> logits(size);
                for (auto &value : logits)
                    value = float(int(rng() % 65536) - 32768) / 65536;
                logits[pending] = 100;
                logits[3] = kind == 0 ? 99 : kind == 1 ? -2 : 1;
                if (kind == 2)
                    for (int i = size - 64; i < size; ++i)
                        logits[i] = 2; // Just outside top64; slow rejection control.
                constexpr int rounds = 32;
                auto measure = [&](bool direct) {
                    std::uint64_t checksum = 0;
                    LogitCopyStats total;
                    const auto start = Clock::now();
                    for (int i = 0; i < rounds; ++i) {
                        logits[100] += float(i & 1 ? 1 : -1) / 65536;
                        std::vector<int> result;
                        LogitCopyStats stats;
                        if (direct)
                            logit_copy_draft(history, pending, 8, logits, result, stats);
                        else
                            result = reference(history, pending, 8, logits, 3, 64);
                        logits[100] -= float(i & 1 ? 1 : -1) / 65536;
                        checksum += result.size();
                        for (int id : result)
                            checksum += id;
                        total.rank_checks += stats.rank_checks;
                        total.logits_examined += stats.logits_examined;
                    }
                    const auto ns =
                        std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
                        rounds;
                    return std::tuple{ns, checksum, total};
                };
                measure(false);
                measure(true);
                for (int trial = 0; trial < 16; ++trial) {
                    decltype(measure(false)) a, b;
                    if (trial % 2 == 0) {
                        a = measure(false);
                        b = measure(true);
                    } else {
                        b = measure(true);
                        a = measure(false);
                    }
                    check(std::get<1>(a) == std::get<1>(b), "component selection differs");
                    std::cout << json({{"membership_backend", logit_copy_backend()},
                                       {"vocabulary", size},
                                       {"history_tokens", history.size() + 1},
                                       {"kind", kind},
                                       {"trial", trial},
                                       {"rounds", rounds},
                                       {"sorted_reference_ns", std::get<0>(a)},
                                       {"membership_ns", std::get<0>(b)},
                                       {"checksum", std::get<1>(a)},
                                       {"rank_checks", std::get<2>(b).rank_checks},
                                       {"logits_examined", std::get<2>(b).logits_examined}})
                                     .dump()
                              << '\n';
                }
            }
}
int main(int argc, char **argv) {
    if (argc == 1)
        semantics();
    else if (argc == 2 && std::string(argv[1]) == "--bench")
        benchmark();
    else if (argc == 3 && std::string(argv[1]) == "--replay")
        replay_file(argv[2]);
    else
        throw std::runtime_error("usage: test_logit_copy [--bench | --replay DATA.jsonl]");
}
