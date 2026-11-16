// Opt-in ordinary CPU diagnostic. No latency metrics or speculative branch data.
#include "nlohmann/json.hpp"
#include "runtime.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
using json = nlohmann::json;
using namespace tokenomics;
static void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    try {
        check(argc == 4, "usage: collect_target_logits MODEL JOBS.jsonl OUT.jsonl");
        check(!std::filesystem::exists(argv[3]), "refusing to overwrite collector output");
        std::ifstream jobs(argv[2]);
        check(bool(jobs), "cannot open collector jobs");
        Engine engine({argv[1], "", 2048, 512, 2, 0});
        std::ofstream output(argv[3]);
        check(bool(output), "cannot open collector output");
        int count = 0;
        for (std::string line; std::getline(jobs, line);) {
            const auto job = json::parse(line);
            Request request;
            request.prompt = job.at("prompt").get<std::string>();
            request.max_tokens = job.at("max_tokens").get<int>();
            request.detailed = false;
            check(request.max_tokens >= 1 && request.max_tokens <= 64, "collector output cap1..64");
            const auto expected = engine.generate(request);
            json rows = json::array();
            request.logit_observer = [&](int index, const float *logits, int size) {
                check(index == int(rows.size()), "nonsequential collector output row");
                check(logits && size >= 64, "collector vocabulary is too small");
                for (int i = 0; i < size; ++i)
                    check(std::isfinite(logits[i]), "nonfinite collector logit");
                std::vector<int> ids(size);
                std::iota(ids.begin(), ids.end(), 0);
                std::partial_sort(ids.begin(), ids.begin() + 64, ids.end(), [&](int a, int b) {
                    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
                });
                ids.resize(64);
                rows.push_back({{"output_index", index}, {"top_ids", ids}});
            };
            const auto actual = engine.generate(request);
            check(actual.tokens == expected.tokens && actual.prompt_ids == expected.prompt_ids &&
                      actual.stop == expected.stop,
                  "observer changed canonical IDs or stopping");
            check(rows.size() == actual.tokens.size(), "missing canonical logit rows");
            for (size_t i = 0; i < rows.size(); ++i)
                check(rows[i]["top_ids"][0].get<int>() == actual.tokens[i],
                      "raw ranking differs from selected token; model suppression unsupported");
            output << json({{"prompt_id", job.at("prompt_id")},
                            {"prompt_ids", actual.prompt_ids},
                            {"token_ids", actual.tokens},
                            {"stop", actual.stop},
                            {"rows", rows},
                            {"observer_exact", true},
                            {"system_info", engine.system_info()}})
                          .dump()
                   << '\n';
            output.flush();
            check(bool(output), "collector output write failed");
            ++count;
        }
        check(count > 0, "no collector jobs");
        std::cout << count << " canonical CPU target ranking requests completed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
