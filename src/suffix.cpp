#include "suffix.h"
#include <algorithm>
#include <stdexcept>

namespace tokenomics {
void suffix_draft(std::span<const int> history, int pending, int cap, std::vector<int> &out) {
    if (cap < 0 || cap > 8)
        throw std::invalid_argument("suffix proposal cap must be in [0,8]");
    out.clear();
    const size_t size = history.size() + 1;
    if (cap == 0 || size <= 5)
        return;
    const auto token = [&](size_t i) { return i < history.size() ? history[i] : pending; };
    const size_t limit = std::min(size_t(32), size - 1);
    size_t best_length = 4, continuation = size;
    // end is the exclusive end of an earlier match. Starting at size-1
    // excludes the current suffix while allowing causal overlapping matches.
    for (size_t end = size - 1; end >= 5; --end) {
        size_t length = 0;
        while (length < std::min(limit, end) && token(end - 1 - length) == token(size - 1 - length))
            ++length;
        if (length > best_length) {
            best_length = length;
            continuation = end;
            if (length == limit)
                break;
        }
    }
    for (size_t i = continuation; i < std::min(size, continuation + size_t(cap)); ++i)
        out.push_back(token(i));
}
} // namespace tokenomics
