#include "controller_io.h"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace tokenomics;
using json = nlohmann::json;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error("check failed: " #x);                                         \
    } while (0)
template <class F> static void invalid(F f) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);
}
int main() {
    ControllerConfig cfg;
    cfg.prior_id = "synthetic-io-test-v1";
    cfg.synthetic = true;
    for (std::size_t i = 0; i < actions.size(); ++i)
        cfg.priors[i] = {double(actions[i] + 1), double(i + 1)};
    const auto good = controller_config_json(cfg);
    CHECK(controller_config_json(parse_controller_config(good)) == good);
    auto metadata = good;
    metadata["provenance"] = {{"binary_sha256", std::string(64, 'a')}, {"split", "development"}};
    CHECK(parse_controller_config(metadata).prior_id == cfg.prior_id);
    for (int kind = 0; kind < 23; ++kind) {
        auto j = good;
        switch (kind) {
        case 0:
            j.erase("prior_id");
            break;
        case 1:
            j["schema_version"] = 2;
            break;
        case 2:
            j["schema_version"] = 1.0;
            break;
        case 3:
            j["synthetic"] = 1;
            break;
        case 4:
            j["unexpected"] = true;
            break;
        case 5:
            j["priors"].erase(0);
            break;
        case 6:
            j["priors"][1]["action"] = 0;
            break;
        case 7:
            j["priors"][1]["action"] = 3;
            break;
        case 8:
            j["priors"][1]["action"] = 1.0;
            break;
        case 9:
            j["priors"][0]["base_ms"] = 0;
            break;
        case 10:
            j["priors"][0]["base_ms"] = std::numeric_limits<double>::quiet_NaN();
            break;
        case 11:
            j["priors"][0]["base_ms"] = true;
            break;
        case 12:
            j["priors"][0]["committed_tokens"] = 2;
            break;
        case 13:
            j["priors"][0]["other"] = 1;
            break;
        case 14:
            j["provenance"] = "unstructured";
            break;
        case 15:
            j["min_hold_cycles"] = std::uint64_t(-1);
            break;
        case 16:
            j["probe_interval"] = 0;
            break;
        case 17:
            j["alpha"] = true;
            break;
        case 18:
            j["enable_probes"] = "false";
            break;
        case 19:
            j["switch_margin"] = -0.1;
            break;
        case 20:
            j["sync_ms_per_token"] = -1;
            break;
        case 21:
            j["priors"][0].erase("action");
            break;
        case 22:
            j = json::array();
            break;
        }
        invalid([&] { parse_controller_config(j); });
    }
    auto policy = make_policy("cost-aware", 4, {}, &cfg);
    const auto action = policy->select_action({0, 10, 10, 0});
    policy->observe({action, 0, 0, 1, 1, 0, 0, action});
    auto trace = controller_trace_json(*policy->trace());
    CHECK(trace.at("candidates").size() == 6);
    CHECK(trace.at("selected") == action && trace.at("learning_applied") == true);
    CHECK(json::parse(trace.dump()) == trace);
    std::cout << "strict controller schema and decision trace checks passed\n";
}
