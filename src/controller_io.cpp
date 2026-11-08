#include "controller_io.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace tokenomics {
using json = nlohmann::json;
static double real(const json &j, const char *key) {
    const auto &value = j.at(key);
    if (!value.is_number())
        throw std::invalid_argument(std::string("controller number required: ") + key);
    return value.get<double>();
}
static int integer(const json &j, const char *key) {
    const auto &value = j.at(key);
    if (!value.is_number_integer() ||
        (value.is_number_unsigned() &&
         value.get<std::uint64_t>() > std::uint64_t(std::numeric_limits<int>::max())))
        throw std::invalid_argument(std::string("controller integer required: ") + key);
    const auto number = value.get<std::int64_t>();
    if (number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max())
        throw std::invalid_argument("controller integer out of range");
    return int(number);
}
ControllerConfig parse_controller_config(const json &j) {
    if (!j.is_object() || integer(j, "schema_version") != 1)
        throw std::invalid_argument("controller schema_version must be 1");
    const std::vector<std::string> keys{
        "schema_version", "prior_id",          "synthetic",         "priors",
        "alpha",          "switch_margin",     "sync_ms_per_token", "min_hold_cycles",
        "probe_interval", "allow_target_only", "enable_probes",     "provenance"};
    for (const auto &entry : j.items())
        if (std::find(keys.begin(), keys.end(), entry.key()) == keys.end())
            throw std::invalid_argument("unknown controller field: " + entry.key());
    ControllerConfig cfg;
    cfg.prior_id = j.at("prior_id").get<std::string>();
    cfg.synthetic = j.at("synthetic").get<bool>();
    if (j.contains("alpha"))
        cfg.alpha = real(j, "alpha");
    if (j.contains("switch_margin"))
        cfg.switch_margin = real(j, "switch_margin");
    if (j.contains("sync_ms_per_token"))
        cfg.sync_ms_per_token = real(j, "sync_ms_per_token");
    if (j.contains("min_hold_cycles"))
        cfg.min_hold_cycles = integer(j, "min_hold_cycles");
    if (j.contains("probe_interval"))
        cfg.probe_interval = integer(j, "probe_interval");
    cfg.allow_target_only = j.value("allow_target_only", true);
    cfg.enable_probes = j.value("enable_probes", true);
    if (j.contains("provenance") && !j.at("provenance").is_object())
        throw std::invalid_argument("controller provenance must be an object");
    const auto &priors = j.at("priors");
    if (!priors.is_array() || priors.size() != actions.size())
        throw std::invalid_argument("controller needs exactly six action priors");
    std::array<bool, 6> seen{};
    for (const auto &p : priors) {
        if (!p.is_object() || p.size() != 3 || !p.contains("committed_tokens") ||
            !p.contains("base_ms"))
            throw std::invalid_argument("prior needs action, committed_tokens and base_ms");
        const auto index = action_index(integer(p, "action"));
        if (seen[index])
            throw std::invalid_argument("duplicate controller action prior");
        seen[index] = true;
        cfg.priors[index] = {real(p, "committed_tokens"), real(p, "base_ms")};
    }
    cfg.validate();
    return cfg;
}
json controller_config_json(const ControllerConfig &cfg) {
    json priors = json::array();
    for (std::size_t i = 0; i < actions.size(); ++i)
        priors.push_back({{"action", actions[i]},
                          {"committed_tokens", cfg.priors[i].committed_tokens},
                          {"base_ms", cfg.priors[i].base_ms}});
    return {{"schema_version", 1},
            {"prior_id", cfg.prior_id},
            {"synthetic", cfg.synthetic},
            {"priors", priors},
            {"alpha", cfg.alpha},
            {"switch_margin", cfg.switch_margin},
            {"sync_ms_per_token", cfg.sync_ms_per_token},
            {"min_hold_cycles", cfg.min_hold_cycles},
            {"probe_interval", cfg.probe_interval},
            {"allow_target_only", cfg.allow_target_only},
            {"enable_probes", cfg.enable_probes}};
}
json controller_trace_json(const DecisionTrace &t) {
    json candidates = json::array();
    for (const auto &s : t.candidates)
        candidates.push_back({{"action", s.action},
                              {"eligible", s.eligible},
                              {"committed_tokens", s.committed_tokens},
                              {"base_ms", s.base_ms},
                              {"transition_ms", s.transition_ms},
                              {"score", s.score}});
    return {{"candidates", candidates},
            {"selected", t.selected},
            {"normal_action", t.normal_action},
            {"probe", t.probe},
            {"normal_switch", t.normal_switch},
            {"forced_tail", t.forced_tail},
            {"reason", decision_reason_name(t.reason)},
            {"predicted_tokens", t.predicted_tokens},
            {"predicted_ms", t.predicted_ms},
            {"predicted_sync_ms", t.predicted_sync_ms},
            {"learning_applied", t.learning_applied},
            {"tail_limited", t.tail_limited},
            {"observed_base_ms", t.observed_base_ms},
            {"estimate_tokens_after", t.estimate_tokens_after},
            {"estimate_base_ms_after", t.estimate_base_ms_after},
            {"sync_ms_per_token_after", t.sync_ms_per_token_after}};
}
} // namespace tokenomics
