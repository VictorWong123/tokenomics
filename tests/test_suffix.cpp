#include "suffix.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

static void check(bool ok) {
    if (!ok)
        throw std::runtime_error("suffix retrieval check failed");
}

// Deliberately independent reference: enumerate lengths then start positions,
// rather than extending backwards from each candidate end as production does.
static std::vector<int> reference(const std::vector<int> &seen, int cap) {
    if (!cap)
        return {};
    for (int length = std::min(32, int(seen.size()) - 1); length >= 5; --length) {
        for (int start = int(seen.size()) - length - 1; start >= 0; --start) {
            if (std::equal(seen.begin() + start, seen.begin() + start + length,
                           seen.end() - length)) {
                const auto end = std::min(seen.size(), size_t(start + length + cap));
                return {seen.begin() + start + length, seen.begin() + end};
            }
        }
    }
    return {};
}

int main() {
    std::mt19937 rng(20261116);
    int cases = 0;
    for (int trial = 0; trial < 1024; ++trial) {
        const int size = 1 + rng() % 1024;
        std::vector<int> seen(size);
        const int alphabet = trial % 3 == 0 ? 1 : (trial % 3 == 1 ? 4 : 256);
        for (auto &token : seen)
            token = rng() % alphabet;
        if (trial % 5 == 0)
            for (int i = 0; i < size; ++i)
                seen[i] = i % 9;
        const auto original = seen;
        for (int cap = 0; cap <= 8; ++cap) {
            std::vector<int> actual{999};
            tokenomics::suffix_draft(std::span(seen).first(seen.size() - 1), seen.back(), cap,
                                     actual);
            check(actual == reference(seen, cap));
            check(seen == original);
            check(actual.size() <= size_t(cap));
            ++cases;
        }
    }
    std::vector<int> out{99};
    tokenomics::suffix_draft({}, 1, 8, out);
    check(out.empty());
    for (int invalid : {-1, 9}) {
        bool threw = false;
        try {
            tokenomics::suffix_draft({}, 1, invalid, out);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw);
    }
    std::cout << cases << " causal suffix retrieval cases passed\n";
}
