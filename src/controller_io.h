#pragma once
#include "nlohmann/json.hpp"
#include "policy.h"
namespace tokenomics {
ControllerConfig parse_controller_config(const nlohmann::json &value);
nlohmann::json controller_config_json(const ControllerConfig &config);
nlohmann::json controller_trace_json(const DecisionTrace &trace);
} // namespace tokenomics
