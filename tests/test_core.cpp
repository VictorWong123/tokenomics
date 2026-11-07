#include "policy.h"
#include "commit.h"
#include <iostream>
#include <stdexcept>
using namespace tokenomics;
#define CHECK(x) do { if(!(x)) throw std::runtime_error("check failed: " #x); } while(0)
int main() {
    CHECK(draft_budget(8,1,200) == 0); CHECK(draft_budget(8,9,3) == 2);
    auto eog = [](int t){return t == 2;};
    auto z = plan_commit(10,5,4,{7},eog); CHECK(z.accepted==0 && z.prefix_after==11);
    auto p = plan_commit(10,5,4,{7,8,9},eog); CHECK(p.accepted==2 && p.prefix_after==13);
    auto f = plan_commit(10,5,4,{7,8,9,10,11},eog); CHECK(f.accepted==4 && f.tokens.size()==5);
    auto eos = plan_commit(10,5,4,{7,2,9,10,11},eog); CHECK(eos.eog && eos.tokens.size()==2 && eos.accepted==2);
    auto trunc = plan_commit(10,1,4,{7,8,9,10,11},eog); CHECK(trunc.tokens.size()==1 && trunc.accepted==1);
    bool threw=false; try {plan_commit(0,0,1,{},eog);} catch(const std::invalid_argument &) {threw=true;} CHECK(threw);
    auto a = make_policy("acceptance",4);
    CHECK(a->select_action({})==2);
    a->observe({2,2,2,3,1}); a->observe({2,2,2,3,1}); CHECK(a->select_action({})==4);
    a->observe({4,4,1,2,1}); CHECK(a->select_action({})==2);
    a->observe({2,0,0,1,1}); CHECK(a->select_action({})==2);
    a->reset(); CHECK(a->select_action({})==2);
    auto s=make_policy("script",4,{0,8,1}); CHECK(s->select_action({})==0); CHECK(s->select_action({})==8);
    s->reset(); CHECK(s->select_action({})==0);
    for(int k : caps) { auto fixed=make_policy("fixed",k); CHECK(fixed->select_action({})==k); }
    std::cout << "core checks passed\n";
}
