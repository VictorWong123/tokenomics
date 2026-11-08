#include "controller_io.h"
#include "nlohmann/json.hpp"
#include "runtime.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
using json = nlohmann::json;
using namespace tokenomics;
static const char *help = R"(Tokenomics: greedy speculative decoding laboratory
  --target FILE [--draft FILE] [--prompt TEXT | --jobs FILE.jsonl]
  --policy target|fixed|confidence|acceptance|cost-aware|script  (default target)
  --controller FILE.json (required frozen priors for cost-aware)
  --k 1|2|4|6|8 --max-tokens N --context N --batch N --threads N
  --gpu-layers N   (default 0; compile with optional Metal/CUDA for offload)
  --cache retain|upstream --drafter model|ngram --confidence P
  --out FILE.jsonl (default stdout; refuses overwrite)
  --cycles FILE.jsonl --script 0,1,4,0,8 (diagnostic transitions only)
Jobs contain prompt, policy, k, max_tokens, cache, drafter, confidence,
detailed, diagnostic_logits, warmup, controller (object), and arbitrary metadata. prefix_tokens supports diagnostic
replay from exact target trace IDs. No stochastic sampling or context shifting.
capture_checkpoints + checkpoint_namespace save diagnostic KV states in this
process; resume_checkpoint restores them. A new ordinary request releases them.
Cost-aware scheduling is experimental; configuration provenance is retained.
)";
static json cycle_json(const Cycle &c) {
    json result = {{"cycle", c.cycle},
                   {"action", c.action},
                   {"actual_drafts", c.actual},
                   {"effective_cap", c.effective_cap},
                   {"accepted", c.accepted},
                   {"committed", c.committed},
                   {"context_tokens", c.context},
                   {"target_cached", c.target_cached},
                   {"draft_cached", c.draft_cached},
                   {"stale_before", c.stale_before},
                   {"synced", c.synced},
                   {"draft_replay_tokens", c.draft_replay_tokens},
                   {"decision_ms", c.decision_ms},
                   {"sync_ms", c.sync_ms},
                   {"draft_ms", c.draft_ms},
                   {"verify_ms", c.verify_ms},
                   {"cache_ms", c.cache_ms},
                   {"wall_ms", c.wall_ms},
                   {"observation_ms", c.observation_ms},
                   {"policy_update_ms", c.policy_update_ms},
                   {"switched", c.switched}};
    if (c.controller)
        result["controller"] = controller_trace_json(*c.controller);
    if (!c.logits.empty()) {
        result["diagnostic_logits"] = json::array();
        for (const auto &probe : c.logits)
            result["diagnostic_logits"].push_back({{"row", probe.row},
                                                   {"output_index", probe.output_index},
                                                   {"top_id", probe.top_id},
                                                   {"second_id", probe.second_id},
                                                   {"top_logit", probe.top_logit},
                                                   {"second_logit", probe.second_logit}});
    }
    return result;
}
static json execution_json(const ExecutionTotals &e) {
    json per_action = json::array();
    for (std::size_t i = 0; i < actions.size(); ++i)
        per_action.push_back({{"action", actions[i]},
                              {"cycles", e.actions[i].cycles},
                              {"committed_tokens", e.actions[i].committed_tokens},
                              {"wall_ms", e.actions[i].wall_ms}});
    return {{"cycles", e.cycles},
            {"drafted", e.drafted},
            {"accepted", e.accepted},
            {"verified_rows", e.verified_rows},
            {"synced_tokens", e.synced_tokens},
            {"draft_replay_tokens", e.draft_replay_tokens},
            {"switches", e.switches},
            {"decision_ms", e.decision_ms},
            {"policy_update_ms", e.policy_update_ms},
            {"sync_ms", e.sync_ms},
            {"draft_ms", e.draft_ms},
            {"verify_ms", e.verify_ms},
            {"cache_ms", e.cache_ms},
            {"cycle_wall_ms", e.cycle_wall_ms},
            {"actions", per_action}};
}
static json controller_totals_json(const ControllerTotals &c) {
    return {{"probes", c.probes},
            {"normal_switches", c.normal_switches},
            {"forced_tail_cycles", c.forced_tail_cycles},
            {"fallback_cycles", c.fallback_cycles},
            {"fallback_tokens", c.fallback_tokens},
            {"learning_updates", c.learning_updates},
            {"tail_limited", c.tail_limited},
            {"probe_ms", c.probe_ms},
            {"fallback_ms", c.fallback_ms},
            {"predicted_sync_ms", c.predicted_sync_ms},
            {"actual_sync_ms", c.actual_sync_ms},
            {"prediction_abs_ms", c.prediction_abs_ms},
            {"prediction_abs_fraction", c.prediction_abs_fraction}};
}
static int number(const std::string &s) {
    size_t end;
    int n = std::stoi(s, &end);
    if (end != s.size())
        throw std::invalid_argument("invalid integer: " + s);
    return n;
}
int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> a;
        const std::vector<std::string> keys{
            "target",     "draft",      "prompt", "jobs",    "policy",     "k",
            "max-tokens", "context",    "batch",  "threads", "gpu-layers", "cache",
            "drafter",    "confidence", "out",    "cycles",  "script",     "controller"};
        for (int i = 1; i < argc; ++i) {
            std::string key(argv[i]);
            if (key == "--help") {
                std::cout << help;
                return 0;
            }
            if (key.rfind("--", 0) != 0 || i + 1 == argc)
                throw std::invalid_argument("expected --option value");
            key = key.substr(2);
            if (std::find(keys.begin(), keys.end(), key) == keys.end() || a.count(key))
                throw std::invalid_argument("unknown/duplicate option: " + key);
            a[key] = argv[++i];
        }
        if (!a.count("target"))
            throw std::invalid_argument("--target is required");
        if (a.count("prompt") && a.count("jobs"))
            throw std::invalid_argument("choose prompt or jobs");
        if (a.count("jobs"))
            for (auto key : {"policy", "k", "max-tokens", "cache", "drafter", "confidence",
                             "script", "controller"})
                if (a.count(key))
                    throw std::invalid_argument("per-request options belong in --jobs records");
        EngineConfig ec;
        ec.target = a["target"];
        ec.draft = a.count("draft") ? a["draft"] : "";
        if (a.count("context"))
            ec.context = number(a["context"]);
        if (a.count("batch"))
            ec.batch = number(a["batch"]);
        if (a.count("threads"))
            ec.threads = number(a["threads"]);
        if (a.count("gpu-layers"))
            ec.gpu_layers = number(a["gpu-layers"]);
        std::vector<json> jobs;
        double controller_load_ms = 0;
        if (a.count("jobs")) {
            std::ifstream f(a["jobs"]);
            if (!f)
                throw std::runtime_error("cannot read jobs");
            std::string line;
            while (std::getline(f, line))
                if (!line.empty())
                    jobs.push_back(json::parse(line));
            if (jobs.empty())
                throw std::runtime_error("empty jobs file");
        } else {
            json job{{"prompt", a.count("prompt") ? a["prompt"] : "Explain speculative decoding."}};
            for (auto key : {"policy", "cache", "drafter"})
                if (a.count(key))
                    job[key] = a[key];
            for (auto key : {"k", "max-tokens"})
                if (a.count(key))
                    job[key == std::string("max-tokens") ? "max_tokens" : key] = number(a[key]);
            if (a.count("confidence"))
                job["confidence"] = std::stod(a["confidence"]);
            if (a.count("script")) {
                std::vector<int> seq;
                std::stringstream ss(a["script"]);
                std::string value;
                while (std::getline(ss, value, ','))
                    seq.push_back(number(value));
                job["script"] = seq;
            }
            if (a.count("controller")) {
                const auto load_start = std::chrono::steady_clock::now();
                std::ifstream config(a["controller"]);
                if (!config)
                    throw std::runtime_error("cannot read controller configuration");
                job["controller"] = json::parse(config);
                controller_load_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - load_start)
                                         .count();
            }
            jobs.push_back(job);
        }
        auto open = [](const std::string &p, std::ofstream &f) {
            if (std::filesystem::exists(p))
                throw std::runtime_error("refusing to overwrite " + p);
            f.open(p);
            if (!f)
                throw std::runtime_error("cannot write " + p);
        };
        std::ofstream rf, cf;
        if (a.count("out"))
            open(a["out"], rf);
        if (a.count("cycles"))
            open(a["cycles"], cf);
        std::ostream &output = rf.is_open() ? rf : std::cout;
        Engine engine(ec);
        for (auto job : jobs) {
            const auto request_start = std::chrono::steady_clock::now();
            Request r;
            r.prompt = job.value("prompt", "");
            r.mode = job.value("policy", "target");
            r.k = job.value("k", 4);
            r.max_tokens = job.value("max_tokens", 128);
            r.max_cycles = job.value("max_cycles", 0);
            r.cache = job.value("cache", "retain");
            r.drafter = job.value("drafter", "model");
            r.confidence = job.value("confidence", 0.5);
            r.detailed = job.value("detailed", true);
            r.diagnostic_logits = job.value("diagnostic_logits", false);
            r.script = job.value("script", std::vector<int>{});
            r.prefix_tokens = job.value("prefix_tokens", std::vector<int>{});
            r.capture_checkpoints = job.value("capture_checkpoints", std::vector<int>{});
            r.checkpoint_namespace = job.value("checkpoint_namespace", "");
            r.resume_checkpoint = job.value("resume_checkpoint", "");
            double controller_parse_ms = 0;
            if (job.contains("controller")) {
                if (r.mode != "cost-aware")
                    throw std::invalid_argument(
                        "controller configuration requires cost-aware policy");
                const auto parse_start = std::chrono::steady_clock::now();
                r.controller = parse_controller_config(job.at("controller"));
                controller_parse_ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - parse_start)
                                          .count();
            }
            auto result = engine.generate(r);
            const auto request_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - request_start)
                                        .count();
            json meta{{"run_id", "adhoc"}, {"configuration_id", r.mode}, {"prompt_id", "adhoc"},
                      {"repetition", 0},   {"output_cap", r.max_tokens}, {"order", 0},
                      {"warmup", false}};
            for (auto key : {"run_id", "configuration_id", "prompt_id", "repetition", "output_cap",
                             "order", "warmup", "checkpoint"})
                if (job.contains(key))
                    meta[key] = job[key];
            meta["schema_version"] = 1;
            if (cf.is_open())
                for (const auto &c : result.cycles) {
                    json cj = cycle_json(c);
                    cj.update(meta);
                    cf << cj.dump() << '\n';
                }
            json rec = meta;
            rec["settings"] = {{"policy", r.mode},
                               {"k", r.k},
                               {"cache", r.cache},
                               {"drafter", r.drafter},
                               {"confidence", r.confidence},
                               {"max_tokens", r.max_tokens},
                               {"context", ec.context},
                               {"batch", ec.batch},
                               {"threads", ec.threads},
                               {"gpu_layers", ec.gpu_layers},
                               {"sampler", "greedy-temperature-only"},
                               {"detailed", r.detailed}};
            rec["settings"]["max_cycles"] = r.max_cycles;
            rec["settings"]["script"] = r.script;
            rec["settings"]["diagnostic_logits"] = r.diagnostic_logits;
            if (r.controller) {
                rec["settings"]["controller"] = controller_config_json(*r.controller);
                if (job.at("controller").contains("provenance"))
                    rec["settings"]["controller"]["provenance"] =
                        job.at("controller").at("provenance");
            }
            rec["settings"]["capture_checkpoints"] = r.capture_checkpoints;
            rec["settings"]["resume_checkpoint"] = r.resume_checkpoint;
            rec["settings"]["diagnostic"] =
                r.diagnostic_logits || r.max_cycles > 0 || !r.prefix_tokens.empty() ||
                !r.capture_checkpoints.empty() || !r.resume_checkpoint.empty();
            rec["checkpoint_ids"] = result.checkpoint_ids;
            rec["checkpoint_bytes"] = result.checkpoint_bytes;
            rec["checkpoint_save_ms"] = result.checkpoint_save_ms;
            rec["checkpoint_restore_ms"] = result.checkpoint_restore_ms;
            rec["llama_revision"] = TOKENOMICS_LLAMA_REV;
            rec["system_info"] = engine.system_info();
            rec["prompt_ids"] = result.prompt_ids;
            rec["token_ids"] = result.tokens;
            // Byte-fallback tokens can end halfway through a UTF-8 code point
            // at a budget/context boundary. Keep IDs authoritative and repair
            // only the display string before strict JSON serialization.
            json display = result.text;
            if (std::any_of(result.text.begin(), result.text.end(),
                            [](unsigned char c) { return c >= 0x80; })) {
                display = json::parse(display.dump(-1, ' ', false, json::error_handler_t::replace));
            }
            rec["text_utf8_repaired"] = display.get_ref<const std::string &>() != result.text;
            rec["text"] = std::move(display);
            rec["stop"] = result.stop;
            rec["load_ms"] = engine.load_ms();
            rec["setup_ms"] = result.setup_ms;
            rec["policy_init_ms"] = result.policy_init_ms;
            rec["controller_parse_ms"] = controller_parse_ms;
            rec["controller_file_load_ms"] = controller_load_ms;
            rec["execution"] = execution_json(result.execution);
            rec["controller_active"] = result.controller_active;
            if (result.controller_active)
                rec["controller"] = controller_totals_json(result.controller);
            rec["prompt_target_ms"] = result.prompt_target_ms;
            rec["prompt_draft_ms"] = result.prompt_draft_ms;
            rec["generation_ms"] = result.generation_ms;
            rec["warm_ms"] = request_ms;
            rec["commit_boundary_ms"] = result.warm_ms;
            rec["first_token_ms"] = result.first_token_ms;
            rec["peak_rss_bytes"] = result.peak_rss_bytes;
            rec["target_cached"] = result.final_target_cached;
            rec["draft_cached"] = result.final_draft_cached;
            rec["generated_tokens"] = result.tokens.size();
            output << rec.dump() << '\n';
            output.flush();
        }
    } catch (const std::exception &ex) {
        std::cerr << "tokenomics: " << ex.what() << '\n';
        return 1;
    }
}
