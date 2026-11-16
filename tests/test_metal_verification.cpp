#include "metal_verification.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

static void check(bool condition) {
    if (!condition)
        throw std::runtime_error("Metal verification split invariant failed");
}

int main() {
    int cases = 0;
    for (int prefix = 0; prefix <= 8192; ++prefix)
        for (unsigned tokens = 1; tokens <= 9; ++tokens)
            for (unsigned configured : {1, 2, 3, 4, 8, 9, 16, 512}) {
                const unsigned chunk = tokenomics::metal_verification_ubatch(
                    configured, prefix, tokens, tokens, true, true, true);
                check(chunk >= 1 && chunk <= configured);
                unsigned consumed = 0;
                while (consumed < tokens) {
                    const unsigned count = std::min(chunk, tokens - consumed);
                    const unsigned extent = (prefix + consumed + count + 255) / 256;
                    for (unsigned row = 0; row < count; ++row)
                        check(extent == (prefix + consumed + row + 1 + 255) / 256);
                    consumed += count;
                }
                check(consumed == tokens);
                ++cases;
            }
    // Prompt/catch-up paths and CPU/noncausal settings stay configured.
    for (unsigned tokens : {1, 8, 9, 10, 512}) {
        check(tokenomics::metal_verification_ubatch(512, 255, tokens, 1, false, true, true) == 512);
        check(tokenomics::metal_verification_ubatch(512, 255, tokens, 1, true, false, true) == 512);
        if (tokens > 1)
            check(tokenomics::metal_verification_ubatch(512, 255, tokens, 1, true, true, true) ==
                  512);
    }
    check(tokenomics::metal_verification_ubatch(512, 254, 9, 9, true, true, true) == 2);
    check(tokenomics::metal_verification_ubatch(512, 255, 9, 9, true, true, true) == 1);
    check(tokenomics::metal_verification_ubatch(512, 256, 9, 9, true, true, true) == 512);
    check(tokenomics::metal_verification_ubatch(512, -1, 9, 9, true, true, true) == 1);
    check(tokenomics::metal_verification_ubatch(512, 256, 9, 9, true, true, false) == 1);
    std::cout << cases
              << " verification splits preserve ordinary padded KV extents and every row\n";
}
