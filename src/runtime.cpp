#include "runtime.h"
#include "commit.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <sys/resource.h>

namespace tokenomics {
static std::mutex evidence_mutex;
static std::vector<std::string> offload_evidence;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static void require(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
long long peak_rss_bytes() {
    rusage r{};
    if (getrusage(RUSAGE_SELF, &r))
        return -1;
#ifdef __APPLE__
    return r.ru_maxrss;
#else
    return r.ru_maxrss * 1024LL;
#endif
}
struct Backend {
    Backend() {
#ifdef TOKENOMICS_CPU_BATCH_INVARIANT
        // The guarded backend uses tiled kernels only above the nine-row
        // verification ceiling. Keep larger common prefills accelerated.
        if (setenv("GGML_CPU_TILED_MM", "1", 1) != 0)
            throw std::runtime_error("cannot select consistent CPU kernels");
#endif
        llama_log_set(
            [](ggml_log_level l, const char *s, void *) {
                if (std::strstr(s, "offloaded ")) {
                    std::lock_guard guard(evidence_mutex);
                    offload_evidence.emplace_back(s);
                }
                if (l == GGML_LOG_LEVEL_ERROR)
                    std::fputs(s, stderr);
            },
            nullptr);
        common_log_set_verbosity_thold(-1);
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~Backend() { llama_backend_free(); }
};
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
static int cached(llama_context *c) {
    return c ? llama_memory_seq_pos_max(llama_get_memory(c), 0) + 1 : 0;
}
static void trim(llama_context *c, int n) {
    require(llama_memory_seq_rm(llama_get_memory(c), 0, n, -1), "partial KV rollback failed");
}
static void decode(llama_context *ctx, const std::vector<int> &ids, int start, int count, int pos,
                   int batch_size) {
    if (count == 0)
        return;
    common_batch b(ctx);
    b.tokens.reserve(std::min(batch_size, count));
    for (int offset = 0; offset < count;) {
        const int len = std::min(batch_size, count - offset);
        b.clear();
        for (int i = 0; i < len; ++i)
            b.add(ids[start + offset + i], pos + offset + i, 0, i == len - 1);
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0,
                "prefill/catch-up decode failed");
        // Every path uses the same completed-work timing convention, including prefill.
        llama_synchronize(ctx);
        offset += len;
    }
}
static void compatible(const llama_model *a, const llama_model *b) {
    require(common_speculative_are_compatible(a, b),
            "upstream tokenizer compatibility check failed");
    const auto *va = llama_model_get_vocab(a);
    const auto *vb = llama_model_get_vocab(b);
    const int n = llama_vocab_n_tokens(va);
    require(n == llama_vocab_n_tokens(vb), "vocabulary sizes differ (strict Tokenomics contract)");
    for (int i = 0; i < n; ++i) {
        require(std::strcmp(llama_vocab_get_text(va, i), llama_vocab_get_text(vb, i)) == 0 &&
                    llama_vocab_get_attr(va, i) == llama_vocab_get_attr(vb, i) &&
                    llama_vocab_is_eog(va, i) == llama_vocab_is_eog(vb, i),
                "vocabulary IDs/attributes/EOG differ");
    }
}
struct Engine::Impl {
    EngineConfig cfg;
    Model target{nullptr, llama_model_free}, draft{nullptr, llama_model_free};
    Context tgt{nullptr, llama_free}, dft{nullptr, llama_free};
    double loading = 0;
    std::string execution_evidence;
    explicit Impl(const EngineConfig &c) : cfg(c) {
        static Backend backend;
        (void)backend;
        require(c.context >= 16 && c.batch >= 9 && c.threads > 0 && c.gpu_layers >= 0,
                "invalid context, batch, threads or gpu_layers");
        require(c.gpu_layers == 0 ||
                    ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
                    ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr,
                "GPU layers requested but no supported local GPU backend is available");
        size_t evidence_start;
        {
            std::lock_guard guard(evidence_mutex);
            evidence_start = offload_evidence.size();
        }
        auto t = Clock::now();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = c.gpu_layers;
        target.reset(llama_model_load_from_file(c.target.c_str(), mp));
        require(bool(target), "cannot load target GGUF");
        if (!c.draft.empty()) {
            draft.reset(llama_model_load_from_file(c.draft.c_str(), mp));
            require(bool(draft), "cannot load draft GGUF");
            compatible(target.get(), draft.get());
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = c.context;
        cp.n_batch = c.batch;
        cp.n_ubatch = c.batch;
        cp.n_seq_max = 1;
        // At most eight proposals plus the pending token; prefill only requests
        // its final row. Apply the same output capacity to every comparison.
        cp.n_outputs_max = 9;
        cp.n_outputs_max_per_seq = 9;
        cp.n_threads = c.threads;
        cp.n_threads_batch = c.threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf = true;
        cp.offload_kqv = c.gpu_layers > 0;
        cp.op_offload = c.gpu_layers > 0;
        tgt.reset(llama_init_from_model(target.get(), cp));
        require(bool(tgt), "cannot initialize target context");
        if (draft) {
            dft.reset(llama_init_from_model(draft.get(), cp));
            require(bool(dft), "cannot initialize draft context");
        }
        require(common_context_can_seq_rm(tgt.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_PART,
                "only full-attention transformer models with arbitrary partial KV removal are "
                "supported");
        if (dft)
            require(common_context_can_seq_rm(dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_PART,
                    "draft lacks arbitrary partial KV removal");
        loading = elapsed(t);
        execution_evidence = "; devices=";
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            execution_evidence += std::string(ggml_backend_dev_name(ggml_backend_dev_get(i))) + ",";
        {
            std::lock_guard guard(evidence_mutex);
            for (size_t i = evidence_start; i < offload_evidence.size(); ++i)
                execution_evidence += "; " + offload_evidence[i];
        }
    }
};
Engine::Engine(const EngineConfig &c) : impl_(std::make_unique<Impl>(c)) {}
Engine::~Engine() = default;
double Engine::load_ms() const { return impl_->loading; }
std::string Engine::system_info() const {
    auto info = std::string(llama_print_system_info()) + impl_->execution_evidence;
#ifdef TOKENOMICS_CPU_BATCH_INVARIANT
    info += "; TOKENOMICS_CPU_BATCH_INVARIANT=ON; GGML_CPU_TILED_MM=1; tiled_min_rows=10";
#endif
    return info;
}

Result Engine::generate(const Request &r) {
    auto &e = *impl_;
    const auto begin = Clock::now();
    require(r.max_tokens >= 0 && r.max_cycles >= 0, "token/cycle budgets must be nonnegative");
    require(r.cache == "retain" || r.cache == "upstream", "cache must be retain or upstream");
    require(r.drafter == "model" || r.drafter == "ngram", "drafter must be model or ngram");
    require(r.confidence >= 0 && r.confidence <= 1, "confidence must be in [0,1]");
    const bool use_model = r.drafter == "model" && r.mode != "target";
    require(!use_model || bool(e.dft), "speculative model mode requires --draft");
    auto policy = make_policy(r.mode, r.k, r.script);
    policy->reset();
    auto *tgt = e.tgt.get();
    auto *dft = e.dft.get();
    const auto *vocab = llama_model_get_vocab(e.target.get());
    llama_memory_clear(llama_get_memory(tgt), true);
    llama_synchronize(tgt);
    if (dft)
        llama_memory_clear(llama_get_memory(dft), true);
    if (dft)
        llama_synchronize(dft);
    Result out;
    out.prompt_ids =
        r.prefix_tokens.empty() ? common_tokenize(vocab, r.prompt, true, true) : r.prefix_tokens;
    require(!out.prompt_ids.empty(), "prompt tokenizes to no tokens; provide nonempty prompt");
    const int context_limit = std::min(e.cfg.context, int(llama_n_ctx(tgt)));
    require(out.prompt_ids.size() <= size_t(context_limit), "prompt exceeds context limit");
    for (int id : out.prompt_ids)
        require(id >= 0 && id < llama_vocab_n_tokens(vocab), "invalid prefix token ID");
    common_params_sampling sampling;
    sampling.temp = 0;
    sampling.seed = 0;
    sampling.no_perf = true;
    sampling.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
    common_sampler_ptr smpl(common_sampler_init(e.target.get(), sampling));
    require(bool(smpl), "sampler initialization failed");
    common_params_speculative sp;
    sp.types = {r.drafter == "model" ? COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE
                                     : COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE};
    sp.draft.ctx_tgt = tgt;
    sp.draft.ctx_dft = dft;
    sp.draft.n_max = 8;
    sp.draft.n_min = 0;
    sp.draft.backend_sampling = false;
    sp.draft.p_min = r.mode == "confidence" ? float(r.confidence) : 0.0f;
    sp.ngram_simple.size_n = 3;
    sp.ngram_simple.size_m = 8;
    sp.ngram_simple.min_hits = 1;
    common_speculative_ptr spec;
    if (r.mode != "target")
        spec.reset(common_speculative_init(sp, 1));
    require(r.mode == "target" || bool(spec), "speculator initialization failed");
    out.setup_ms = elapsed(begin);
    if (r.max_tokens == 0) {
        out.stop = "budget";
        out.warm_ms = elapsed(begin);
        out.peak_rss_bytes = peak_rss_bytes();
        return out;
    }
    // Both target-only and speculative paths keep exactly one pending token.
    std::vector<int> prefix(out.prompt_ids.begin(), out.prompt_ids.end() - 1);
    int pending = out.prompt_ids.back();
    int n = int(prefix.size());
    const int capacity = std::min(r.max_tokens, context_limit - n);
    prefix.reserve(prefix.size() + capacity);
    out.tokens.reserve(capacity);
    if (r.detailed)
        out.cycles.reserve(capacity);
    auto t = Clock::now();
    decode(tgt, prefix, 0, n, 0, e.cfg.batch);
    out.prompt_target_ms = elapsed(t);
    // Diagnostic replay must start every action, including action 0, from the
    // same two-model KV state. Ordinary target-only requests do not do this.
    const bool replay_resident = !r.prefix_tokens.empty() && r.drafter == "model" && dft != nullptr;
    if (use_model || replay_resident) {
        t = Clock::now();
        decode(dft, prefix, 0, n, 0, e.cfg.batch);
        out.prompt_draft_ms = elapsed(t);
    }
    if (spec)
        common_speculative_begin(spec.get(), 0, prefix);
    // Reuse the bounded proposal and verification buffers across all cycles.
    // common_batch::clear preserves its allocation at the pinned revision.
    common_batch batch(tgt);
    batch.tokens.reserve(9);
    std::vector<int> proposed;
    proposed.reserve(8);
    const auto gen_start = Clock::now();
    int previous = -1;
    int cycle_count = 0;
    while (int(out.tokens.size()) < r.max_tokens && n < context_limit) {
        Cycle c;
        c.cycle = cycle_count++;
        c.context = n + 1;
        const int remaining = r.max_tokens - int(out.tokens.size());
        const auto cycle_start = Clock::now();
        t = Clock::now();
        c.stale_before = use_model ? n - cached(dft) : 0;
        c.action = policy->select_action({c.cycle, n + 1, remaining, c.stale_before});
        require(valid_action(c.action), "policy selected unsupported action");
        c.switched = previous >= 0 && previous != c.action;
        previous = c.action;
        c.decision_ms = elapsed(t);
        const int cap = draft_budget(c.action, remaining, context_limit - n);
        proposed.clear();
        if (cap > 0) {
            if (use_model) {
                t = Clock::now();
                const int dpos = cached(dft);
                require(dpos <= n, "draft cache ahead of committed target prefix");
                c.synced = n - dpos;
                decode(dft, prefix, dpos, c.synced, dpos, e.cfg.batch);
                c.sync_ms = elapsed(t);
            }
            t = Clock::now();
            auto &dp = common_speculative_get_draft_params(spec.get(), 0);
            dp = {};
            dp.drafting = true;
            dp.n_max = cap;
            dp.pos0 = n;
            dp.id_last = pending;
            dp.prompt = &prefix;
            dp.result = &proposed;
            common_speculative_draft(spec.get());
            if (use_model)
                llama_synchronize(dft);
            require(
                !use_model || r.mode == "confidence" || !proposed.empty(),
                "unconditional upstream drafter produced no proposal (possible execution failure)");
            require(int(proposed.size()) <= cap, "upstream exceeded requested draft cap");
            if (r.proposal_transform)
                r.proposal_transform(proposed, c.cycle);
            require(int(proposed.size()) <= cap, "test proposal exceeded cap");
            c.draft_ms = elapsed(t);
            // The upstream-style control discards all speculative draft KV, then
            // reprocesses the full target verification batch, including rejected suffixes.
            if (use_model && r.cache == "upstream") {
                t = Clock::now();
                trim(dft, n);
                c.cache_ms += elapsed(t);
            }
        }
        c.actual = int(proposed.size());
        batch.clear();
        batch.add(pending, n, 0, true);
        for (size_t i = 0; i < proposed.size(); ++i)
            batch.add(proposed[i], n + 1 + i, 0, true);
        t = Clock::now();
        require(llama_process(tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                "target verification decode failed");
        if (r.diagnostic_logits) {
            // Observe raw target rows without changing the upstream sampler.
            // Rows beyond the committed prefix describe hypothetical drafts.
            const int size = llama_vocab_n_tokens(vocab);
            require(size >= 2, "logit diagnostic requires at least two tokens");
            for (int row = 0; row <= c.actual; ++row) {
                const float *values = llama_get_logits_ith(tgt, row);
                require(values != nullptr, "missing diagnostic logits");
                int best = -1, second = -1;
                for (int id = 0; id < size; ++id) {
                    require(std::isfinite(values[id]), "nonfinite diagnostic logit");
                    if (best < 0 || values[id] > values[best]) {
                        second = best;
                        best = id;
                    } else if (second < 0 || values[id] > values[second]) {
                        second = id;
                    }
                }
                c.logits.push_back({row, int(out.tokens.size()) + row, best, second, values[best],
                                    values[second]});
            }
        }
        // Upstream sampler owns verification, including correction and bonus.
        const auto ids = common_sampler_sample_and_accept_n(smpl.get(), tgt, proposed);
        c.verify_ms = elapsed(t); // reading logits synchronizes target backend work
        t = Clock::now();
        if (use_model && r.cache == "upstream" && cap > 0) {
            require(common_speculative_process(spec.get(), batch), "upstream draft process failed");
            llama_synchronize(dft);
        }
        auto plan = plan_commit(n, remaining, c.actual, ids,
                                [vocab](int id) { return llama_vocab_is_eog(vocab, id); });
        if (spec && cap > 0)
            common_speculative_accept(spec.get(), 0, plan.accepted);
        for (int id : plan.tokens) {
            prefix.push_back(pending);
            pending = id;
            out.tokens.push_back(id);
        }
        n = plan.prefix_after;
        trim(tgt, n);
        if (use_model)
            trim(dft, n);
        c.target_cached = cached(tgt);
        c.draft_cached = dft ? cached(dft) : 0;
        require(c.target_cached == n && (!use_model || c.draft_cached <= n),
                "KV position invariant violated");
        require(int(prefix.size()) == n && plan.tokens.size() <= size_t(remaining),
                "commit count invariant violated");
        c.accepted = plan.accepted;
        c.committed = int(plan.tokens.size());
        c.cache_ms += elapsed(t);
        c.wall_ms = elapsed(cycle_start);
        policy->observe({c.action, c.actual, c.accepted, c.committed, c.wall_ms});
        c.wall_ms = elapsed(cycle_start); // include policy-update work in the diagnostic cycle
        if (out.tokens.size() == plan.tokens.size())
            out.first_token_ms = elapsed(begin);
        if (r.detailed)
            out.cycles.push_back(std::move(c)); // serialize after generation
        if (plan.eog) {
            out.stop = "eog";
            break;
        }
        if (r.max_cycles > 0 && cycle_count >= r.max_cycles) {
            out.stop = "checkpoint";
            break;
        }
    }
    out.generation_ms = elapsed(gen_start);
    out.warm_ms = elapsed(begin);
    if (out.stop.empty())
        out.stop = int(out.tokens.size()) == r.max_tokens ? "budget" : "context";
    out.final_target_cached = cached(tgt);
    out.final_draft_cached = dft ? cached(dft) : 0;
    out.peak_rss_bytes = peak_rss_bytes();
    for (int id : out.tokens)
        if (!llama_vocab_is_eog(vocab, id))
            out.text += common_token_to_piece(vocab, id);
    return out;
}
} // namespace tokenomics
