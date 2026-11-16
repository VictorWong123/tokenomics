bool tokenomics_common_enable_direct(common_sampler * sampler) {
    if (!sampler) {
        return false;
    }
    const auto & params = sampler->params;
    sampler->tokenomics_direct_calls = 0;
    // A model suppression/bias stage makes the chain longer and falls back to
    // the existing no-temperature path without bypassing token restrictions.
    sampler->tokenomics_direct =
        !sampler->grmr && !sampler->rbudget && params.grammar.empty() &&
        params.temp == 0.0f && params.dynatemp_range == 0.0f &&
        params.n_probs == 0 && params.no_perf && !params.backend_sampling &&
        llama_sampler_chain_n(sampler->chain) == 1 &&
        std::strcmp(llama_sampler_name(llama_sampler_chain_get(sampler->chain, 0)), "greedy") == 0;
    return sampler->tokenomics_direct;
}

bool tokenomics_common_direct_enabled(const common_sampler * sampler) {
    return sampler && sampler->tokenomics_direct;
}

std::uint64_t tokenomics_common_direct_calls(const common_sampler * sampler) {
    return sampler ? sampler->tokenomics_direct_calls : 0;
}
