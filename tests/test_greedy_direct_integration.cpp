#include "common.h"
#include "greedy_direct_sampling.h"
#include "llama.h"
#include "sampling.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error("direct sampler contract failed: " #x);                       \
    } while (0)

static common_sampler_ptr make(const llama_model *model, int kind) {
    common_params_sampling params;
    params.temp = 0;
    params.no_perf = true;
    params.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE};
    if (kind == 1)
        params.n_probs = 1;
    if (kind == 2)
        params.backend_sampling = true;
    if (kind == 3)
        params.no_perf = false;
    if (kind == 4)
        params.logit_bias.push_back({259, -INFINITY});
    if (kind == 5)
        params.dynatemp_range = 1;
    if (kind == 6)
        params.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= \"a\""};
    common_sampler_ptr sampler(common_sampler_init(model, params));
    CHECK(sampler);
    auto *chain = common_sampler_get(sampler.get());
    // Kind 7 deliberately keeps the redundant filter and must be ineligible.
    if (kind != 7) {
        const int n = llama_sampler_chain_n(chain);
        CHECK(n >= 2);
        CHECK(std::string(llama_sampler_name(llama_sampler_chain_get(chain, n - 2))) == "temp-ext");
        llama_sampler_free(llama_sampler_chain_remove(chain, n - 2));
    }
    return sampler;
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file((std::string(argv[1]) + "/target.gguf").c_str(), mp),
        llama_model_free);
    CHECK(model);
    CHECK(!tokenomics_common_enable_direct(nullptr));
    CHECK(!tokenomics_common_direct_enabled(nullptr));
    CHECK(tokenomics_common_direct_calls(nullptr) == 0);
    int cases = 0;
    for (int kind = 0; kind < 8; ++kind) {
        auto sampler = make(model.get(), kind);
        CHECK(!tokenomics_common_direct_enabled(sampler.get()));
        CHECK(tokenomics_common_enable_direct(sampler.get()) == (kind == 0));
        CHECK(tokenomics_common_direct_calls(sampler.get()) == 0);
        if (kind == 0) {
            common_sampler_ptr cloned(common_sampler_clone(sampler.get()));
            CHECK(!tokenomics_common_direct_enabled(cloned.get()));
            auto copied = make(model.get(), 0);
            CHECK(tokenomics_common_enable_direct(copied.get()));
            common_sampler_copy(sampler.get(), copied.get());
            CHECK(!tokenomics_common_direct_enabled(copied.get()));
            CHECK(tokenomics_common_direct_calls(copied.get()) == 0);
        }
        ++cases;
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> suppressed(
        llama_model_load_from_file((std::string(argv[1]) + "/suppressed.gguf").c_str(), mp),
        llama_model_free);
    CHECK(suppressed);
    auto sampler = make(suppressed.get(), 0);
    CHECK(!tokenomics_common_enable_direct(sampler.get()));
    ++cases;
    sampler.reset();
    suppressed.reset();
    model.reset();
    llama_backend_free();
    std::cout << cases << " direct sampler eligibility cases passed; clone/copy stay disabled\n";
}
