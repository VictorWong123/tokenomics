#pragma once
#include "policy.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace tokenomics {
struct EngineConfig {
    std::string target, draft;
    int context = 2048, batch = 512, threads = 4, gpu_layers = 0;
};
struct Request {
    std::string prompt, mode = "target", cache = "retain", drafter = "model";
    int max_tokens = 128, k = 4;
    int max_cycles = 0; // nonzero only for one-cycle diagnostic replay
    double confidence = 0.5;
    bool detailed = true;
    bool diagnostic_logits = false; // opt-in correctness probe; never benchmark timing
    std::vector<int> script;
    // Test-only fault injection after upstream drafting. Never exposed by CLI.
    std::function<void(std::vector<int> &, int)> proposal_transform;
    // Diagnostic common-prefix replay; authoritative token IDs, no re-tokenization.
    std::vector<int> prefix_tokens;
    // Diagnostic, same-Engine checkpoints. A new ordinary request releases them.
    std::vector<int> capture_checkpoints;
    std::string checkpoint_namespace, resume_checkpoint;
};
struct LogitProbe {
    int row = 0, output_index = 0, top_id = 0, second_id = 0;
    float top_logit = 0, second_logit = 0;
};
struct Cycle {
    int cycle = 0, action = 0, actual = 0, accepted = 0, committed = 0;
    int context = 0, target_cached = 0, draft_cached = 0, stale_before = 0, synced = 0;
    double decision_ms = 0, sync_ms = 0, draft_ms = 0, verify_ms = 0, cache_ms = 0, wall_ms = 0;
    bool switched = false;
    std::vector<LogitProbe> logits;
};
struct Result {
    std::vector<int> prompt_ids, tokens;
    std::vector<Cycle> cycles;
    std::string stop, text;
    double setup_ms = 0, prompt_target_ms = 0, prompt_draft_ms = 0;
    double generation_ms = 0, warm_ms = 0, first_token_ms = 0;
    long long peak_rss_bytes = 0;
    int final_target_cached = 0, final_draft_cached = 0;
    std::vector<std::string> checkpoint_ids;
    std::size_t checkpoint_bytes = 0;
    double checkpoint_save_ms = 0, checkpoint_restore_ms = 0;
};
class Engine {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit Engine(const EngineConfig &);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    Result generate(const Request &);
    double load_ms() const;
    std::string system_info() const;
};
long long peak_rss_bytes();
} // namespace tokenomics
