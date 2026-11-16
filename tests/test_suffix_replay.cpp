// Replay production proposals against canonical IDs; this does not run a model.
#include "nlohmann/json.hpp"
#include "suffix.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("usage: test_suffix_replay JOBS.jsonl");
        std::ifstream input(argv[1]);
        if (!input)
            throw std::runtime_error("cannot open canonical replay jobs");
        std::string line;
        int checked = 0;
        while (std::getline(input, line)) {
            if (line.empty())
                continue;
            const auto job = json::parse(line);
            auto prefix = job.at("prompt_ids").get<std::vector<int>>();
            const auto output = job.at("token_ids").get<std::vector<int>>();
            const int cap = job.at("cap");
            if (prefix.empty() || output.empty() || cap < 1 || cap > 8)
                throw std::runtime_error("invalid replay prompt/output/cap");
            int pending = prefix.back();
            prefix.pop_back();
            int index = 0, cycles = 0, drafted = 0, accepted = 0;
            std::vector<int> proposed;
            while (index < int(output.size())) {
                const int effective = std::min(cap, int(output.size()) - index - 1);
                tokenomics::suffix_draft(prefix, pending, effective, proposed);
                if (int(proposed.size()) > effective)
                    throw std::runtime_error("production suffix exceeded replay budget");
                int matches = 0;
                while (matches < int(proposed.size()) &&
                       proposed[matches] == output[index + matches])
                    ++matches;
                ++cycles;
                drafted += int(proposed.size());
                accepted += matches;
                // Only the committed canonical prefix enters future retrieval.
                for (int step = 0; step <= matches; ++step) {
                    prefix.push_back(pending);
                    pending = output[index++];
                }
            }
            json result = {{"source_id", job.at("source_id")},
                           {"prompt_id", job.at("prompt_id")},
                           {"output_cap", job.at("output_cap")},
                           {"tokens", output.size()},
                           {"cycles", cycles},
                           {"drafted", drafted},
                           {"accepted", accepted},
                           {"verified_rows", cycles + drafted},
                           {"wasted_rows", cycles + drafted - int(output.size())}};
            for (auto it = job.at("expected").begin(); it != job.at("expected").end(); ++it)
                if (!result.contains(it.key()) || result.at(it.key()) != it.value())
                    throw std::runtime_error(
                        "native suffix replay differs from frozen Python counts");
            std::cout << result.dump() << '\n';
            ++checked;
        }
        if (!checked)
            throw std::runtime_error("no canonical replay jobs");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
