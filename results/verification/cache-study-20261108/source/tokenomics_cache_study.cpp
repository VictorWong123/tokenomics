#include "runtime.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tokenomics;
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  Engine engine({argv[2], "", 512, 512, std::stoi(argv[3]), 0});
  Request base;
  for (int i = 0; i < 27; ++i)
    base.prefix_tokens.push_back(3 + 'a' + i % 9);
  base.max_tokens = 64;
  const auto reference = engine.generate(base);
  if (reference.tokens.empty())
    throw std::runtime_error("empty canonical trace");
  for (size_t i = 0; i < reference.tokens.size(); ++i)
    if (reference.tokens[i] != 3 + 'a' + int(i % 9))
      throw std::runtime_error(
          "fixture does not produce the declared period-nine trace");
  std::cout << "# SYNTHETIC PERFECT-COPY KERNEL DIAGNOSTIC; no language-model "
               "speedup claim\n";
  std::cout << "# " << engine.system_info() << '\n';
  std::cout << "profile\tthreads\trepetition\taction\ttokens\tgeneration_"
               "ms\tdrafted\taccepted\tverify_ms\n";
  std::cout << std::setprecision(12);
  std::vector<int> order = {0, 4, 8};
  for (int rep = 0; rep < 3; ++rep) {
    for (int k : order) {
      Request job = base;
      job.max_tokens = int(reference.tokens.size());
      if (k > 0) {
        job.mode = "fixed";
        job.drafter = "ngram";
        job.k = k;
      }
      const auto result = engine.generate(job);
      if (result.tokens != reference.tokens || result.stop != reference.stop)
        throw std::runtime_error(
            "copy batch shape changed canonical tokens/stop");
      int drafted = 0, accepted = 0;
      double verify = 0;
      for (const auto &cycle : result.cycles) {
        drafted += cycle.actual;
        accepted += cycle.accepted;
        verify += cycle.verify_ms;
      }
      if (drafted != accepted)
        throw std::runtime_error("copy proposals not fully accepted");
      std::cout << argv[1] << '\t' << argv[3] << '\t' << rep << '\t' << k
                << '\t' << result.tokens.size() << '\t' << result.generation_ms
                << '\t' << drafted << '\t' << accepted << '\t' << verify
                << '\n';
    }
    std::rotate(order.begin(), order.begin() + 1, order.end());
  }
}
