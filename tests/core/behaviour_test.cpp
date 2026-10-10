#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/mtp_sampling.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace s=gufo::sampling;namespace q=gufo::models::qwen38_flash_next;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(#x)+" line="+std::to_string(__LINE__));}while(false)
std::shared_ptr<s::BehaviourConfig> Config(){auto b=std::make_shared<s::BehaviourConfig>();b->eog_ids={0,1};b->think_start={10};b->think_end={11,12};b->budget_forced={11,12};auto p=std::make_shared<std::vector<std::string>>(256,"x");(*p)[0]="";(*p)[1]="";(*p)[2]="word";(*p)[3]=".";(*p)[10]="<think>";(*p)[11]="</";(*p)[12]="think>";b->token_pieces=p;return b;}
s::SamplingConfig Sampling(std::shared_ptr<s::BehaviourConfig> b,float t=0){s::SamplingConfig c;c.seed=42;c.temperature=t;c.behaviour=std::move(b);return c;}
bool Count(const s::SamplerState& sampler,std::string_view name,unsigned n){unsigned total=0;bool seen=false;for(const auto& l:sampler.BehaviourSummary())if(l.find("name="+std::string(name)+" ")!=std::string::npos){seen=true;const auto at=l.find("taken=")+6;total+=static_cast<unsigned>(std::stoul(l.substr(at)));}return seen&&total==n;}
void BoundaryPort(){ // O/tests/test-eog-boundary.cpp:16, split fences/math/UTF8.
 auto at=[](std::string_view t){s::EogBoundaryState b;b.accept_piece(t);return b.allowed();};
 for(auto t:{"Done.","Done!","Done?","Proof:","Case;","End.  ","Line\n","(x)","[x]","{x}","<em>text</em>","\\(x\\)","\\[x\\]","$x$"})CHECK(at(t));
 for(auto t:{"altern","ordinary word","after,","a >","'","\"","*","_","`code","$x","\\(x","```\ncode\n"})CHECK(!at(t));
 s::EogBoundaryState b;b.accept_piece("`");b.accept_piece("``\ninside.\n");CHECK(!b.allowed());b.accept_piece("``");CHECK(!b.allowed());b.accept_piece("`");CHECK(b.allowed());
 b.reset();b.accept_piece("hello ");b.accept_piece("\xE3\x80");CHECK(!b.allowed());b.accept_piece("\x82");CHECK(b.allowed());auto copy=b;b.accept_piece(" next");CHECK(!b.allowed());CHECK(copy.allowed());
}
void LoopPort(){ // O/tests/test-loop-breaker.cpp:13-103, escalation/copy.
 s::LoopBreaker b;for(int i=0;i<127;++i)CHECK(b.accept(7,s::LoopBreaker::THINK).what==s::LoopBreaker::NONE);CHECK(b.accept(7,s::LoopBreaker::THINK).what==s::LoopBreaker::NUDGE);
 for(int i=0;i<15;++i)CHECK(b.accept(7,s::LoopBreaker::THINK).what==s::LoopBreaker::NONE);CHECK(b.accept(7,s::LoopBreaker::THINK).what==s::LoopBreaker::FORCE_THINK);CHECK(b.take_event().what==s::LoopBreaker::NUDGE);CHECK(b.take_event().what==s::LoopBreaker::FORCE_THINK);
 s::LoopBreaker p;p.cfg={4,12,4,12};for(int i=0;i<11;++i)p.accept(20+i%3,s::LoopBreaker::THINK);CHECK(p.accept(22,s::LoopBreaker::THINK).period==3);
 s::LoopBreaker sixteen;for(int i=0;i<255;++i)sixteen.accept(100+i%16,s::LoopBreaker::THINK);CHECK(sixteen.accept(115,s::LoopBreaker::THINK).period==16);
 s::LoopBreaker answer;for(int i=0;i<127;++i)answer.accept(4,s::LoopBreaker::THINK);for(int i=0;i<511;++i)CHECK(answer.accept(4,s::LoopBreaker::ANSWER).what==s::LoopBreaker::NONE);CHECK(answer.accept(4,s::LoopBreaker::ANSWER).what==s::LoopBreaker::NUDGE);auto copy=answer;for(int i=0;i<16;++i)answer.accept(4,s::LoopBreaker::ANSWER);CHECK(answer.terminal);CHECK(!copy.terminal);
 s::LoopBreaker expired;expired.cfg={4,4,4,4};for(int i=0;i<4;++i)expired.accept(5,s::LoopBreaker::THINK);for(int i=0;i<273;++i)expired.accept(100+i,s::LoopBreaker::THINK);for(int i=0;i<3;++i)expired.accept(5,s::LoopBreaker::THINK);CHECK(expired.accept(5,s::LoopBreaker::THINK).what==s::LoopBreaker::FORCE_THINK);
}
void LoopNearMissRecurrence(){s::LoopBreaker near;for(int i=0;i<255;++i)CHECK(near.accept(100+i%16,s::LoopBreaker::THINK).what==s::LoopBreaker::NONE);CHECK(near.accept(999,s::LoopBreaker::THINK).what==s::LoopBreaker::NONE);for(int i=0;i<200;++i)CHECK(near.accept(1000+i,s::LoopBreaker::THINK).what==s::LoopBreaker::NONE);s::LoopBreaker recurrent;recurrent.cfg={4,4,4,4};for(int i=0;i<4;++i)recurrent.accept(5,s::LoopBreaker::THINK);for(int i=0;i<16;++i)recurrent.accept(100+i,s::LoopBreaker::THINK);for(int i=0;i<3;++i)recurrent.accept(5,s::LoopBreaker::THINK);CHECK(recurrent.accept(5,s::LoopBreaker::THINK).what==s::LoopBreaker::FORCE_THINK);}
void UnsetIdentity(){s::SamplingConfig c;c.temperature=1;c.top_k=3;c.seed=22;auto c2=c;c2.behaviour=Config();s::SamplerState a(c),b(c2);std::vector<float> row={1,2,3,4,5,6};for(int i=0;i<100;++i){auto x=a.Sample(row),y=b.Sample(row);CHECK(x==y);CHECK(a.rng_state()==b.rng_state());a.Accept(x);b.Accept(y);}CHECK(b.BehaviourSummary().empty());}
void GuardBudgetBias(){
 auto b=Config();b->think_eog_guard=true;b->initial_thinking=true;s::SamplerState guard(Sampling(b));std::vector<float> row(16,-10);row[0]=9;row[2]=8;CHECK(guard.Sample(row)==2);CHECK(Count(guard,"think_eog_guard",1));row[0]=-9;CHECK(guard.Sample(row)==2);CHECK(Count(guard,"think_eog_guard",1));guard.Accept(11);row[0]=9;CHECK(guard.Sample(row)==2);guard.Accept(12);CHECK(guard.Sample(row)==0);
 auto z=Config();z->initial_thinking=true;z->reasoning_budget_enabled=true;z->reasoning_budget=0;s::SamplerState budget(Sampling(z));CHECK(budget.Sample(row)==11);budget.Accept(11);CHECK(budget.Sample(row)==12);budget.Accept(12);CHECK(budget.Sample(row)==0);
 auto rescue=Config();rescue->initial_thinking=true;rescue->think_eog_guard=true;rescue->reasoning_budget_enabled=true;rescue->reasoning_budget=0;s::SamplerState rescued(Sampling(rescue));std::vector<float> forced_only(16,-INFINITY);forced_only[0]=10;CHECK(rescued.Sample(forced_only)==11);CHECK(Count(rescued,"reasoning_budget",0));bool diagnostic=false;for(const auto& line:rescued.BehaviourSummary())if(line.find("name=counterfactual taken=0 fallback=1 reason=reasoning_budget_no_legal_token")!=std::string::npos)diagnostic=true;CHECK(diagnostic);
 auto limited=Config();limited->initial_thinking=true;limited->reasoning_budget_enabled=true;limited->reasoning_budget=1;auto pieces=std::make_shared<std::vector<std::string>>(*limited->token_pieces);(*pieces)[2]="\xE3\x80";(*pieces)[3]="\x82";limited->token_pieces=pieces;s::SamplerState utf8(Sampling(limited));utf8.Accept(2);row[3]=10;CHECK(utf8.Sample(row)==3);utf8.Accept(3);CHECK(utf8.Sample(row)==11);
 auto m=Config();m->initial_thinking=true;m->marker_ids={2};m->marker_bias=-2;s::SamplerState marker(Sampling(m));row.assign(16,-10);row[2]=5;row[3]=4;CHECK(marker.Sample(row)==3);CHECK(Count(marker,"marker_bias",1));marker.Accept(11);marker.Accept(12);CHECK(marker.Sample(row)==2);
 auto e=Config();e->logit_bias_enabled=true;e->logit_bias={{3,5}};s::SamplerState explicit_bias(Sampling(e));CHECK(explicit_bias.Sample(row)==3);CHECK(Count(explicit_bias,"logit_bias",1));
}
void MinProbDefer(){ // O/tests/test-eog-min-prob.cpp:9, exemption/spare.
 auto b=Config();b->eog_min_prob=.06;b->eog_min_prob_exact=true;std::vector<float> row={std::log(.052F),std::log(.005F),std::log(.943F)};s::SamplerState exact(Sampling(b,1));auto d=exact.Distribution(row);CHECK(d.probability(0)==0);CHECK(d.probability(1)==0);CHECK(d.probability(2)==1);
 b->logit_bias_enabled=true;b->logit_bias={{0,10}};s::SamplerState exempt(Sampling(b));CHECK(exempt.Sample(row)==0);
 auto g=Config();g->eog_boundary=true;g->eog_defer_p=.9;s::SamplerState deferred(Sampling(g));deferred.Accept(2);row={10,-10,1,0};CHECK(deferred.Sample(row)==2);deferred.Accept(3);row[0]=-10;CHECK(deferred.Sample(row)==0);CHECK(Count(deferred,"eog_defer",1));deferred.ResetHistory({});CHECK(Count(deferred,"eog_defer",0));
 row={10,-10,1,0};auto e=Config();e->eog_boundary=true;e->think_eog_guard=true;e->logit_bias_enabled=true;e->logit_bias={{0,1}};s::SamplerState explicit_eog(Sampling(e));explicit_eog.Accept(2);CHECK(explicit_eog.Sample(row)==0);explicit_eog.Accept(10);CHECK(explicit_eog.Sample(row)!=0);
}
void CandidateProof(){
 auto b=Config();b->think_eog_guard=true;b->initial_thinking=true;auto c=Sampling(b,1);c.top_k=2;s::SamplerState sampler(c);std::vector<float> full={10,9,8,7,6,5},values={10,9,8,7,6};std::vector<s::TokenId> ids={0,1,2,3,4};auto top=sampler.DistributionFromTop(values,ids,6);CHECK(top.has_value());auto all=sampler.Distribution(full);for(unsigned i=0;i<6;++i)CHECK(std::abs(top->probability(i)-all.probability(i))<1e-12);
 std::vector<float> short_values={10,9,8};std::vector<s::TokenId> short_ids={0,1,2};CHECK(!sampler.DistributionFromTop(short_values,short_ids,6));CHECK(Count(sampler,"candidate_path",0));
 auto bias=Config();bias->logit_bias_enabled=true;bias->logit_bias={{5,100}};c.behaviour=bias;s::SamplerState positive(c);CHECK(!positive.DistributionFromTop(values,ids,6));CHECK(positive.Sample(full)==5);
 auto forced=Config();forced->initial_thinking=true;forced->reasoning_budget_enabled=true;forced->reasoning_budget=0;c.behaviour=forced;s::SamplerState outside(c);CHECK(!outside.DistributionFromTop(values,ids,16));
 q::MtpCandidateLogits list;list.size=3;list.ids[0]=200;list.ids[1]=5;list.ids[2]=90;list.logits[0]=3;list.logits[1]=2;list.logits[2]=1;std::uint64_t rng=42;auto proposal=q::SampleMtpProposal(list,positive,&rng);double total=0;for(std::size_t i=0;i<proposal.size;++i){total+=proposal.probabilities[i];CHECK(proposal.ids[i]==200||proposal.ids[i]==5||proposal.ids[i]==90);}CHECK(total==1);
}
void CloseReplay(){auto b=Config();b->initial_thinking=true;b->loop_breaker=true;b->loop_transition=2;b->loop_eog_close=true;b->loop_prefix={4,5};s::SamplerState sampler(Sampling(b));for(int i=0;i<144;++i)sampler.Accept(2);std::vector<float> row(16,-10);row[0]=10;row[2]=9;CHECK(sampler.Sample(row)==4);sampler.Accept(4);CHECK(sampler.Sample(row)==5);sampler.Accept(5);auto before=sampler,working=sampler;auto d=working.Distribution(row);CHECK(d.probability(0)==0);CHECK(d.probability(11)==1);auto token=working.Sample(row);CHECK(token==11);working.Accept(token);sampler.CopyDrawStateFrom(working);sampler.Accept(token);CHECK(Count(sampler,"loop_eog_close",1));CHECK(sampler.Sample(row)==12);CHECK(Count(sampler,"loop_eog_close",2));sampler.Accept(12);CHECK(sampler.Sample(row)==0);CHECK(before.Sample(row)==11);}
void SpecExactness(){auto b=Config();b->think_eog_guard=true;b->initial_thinking=true;s::SamplerState sampler(Sampling(b,1));std::vector<float> row={2,1,0};q::MtpProposal p;p.size=1;p.ids[0]=0;p.probabilities[0]=1;p.token=0;p.probability=1;auto rejected=q::VerifyMtpProposal(row,p,sampler);CHECK(!rejected.accepted);CHECK(rejected.token==2);CHECK(Count(sampler,"think_eog_guard",1));s::SamplerState native(Sampling(Config(),1));std::vector<float> point={10,-INFINITY,-INFINITY};auto accepted=q::VerifyMtpProposal(point,p,native);CHECK(accepted.accepted&&accepted.token==0);p.ids[0]=2;p.token=2;auto residual=q::VerifyMtpProposal(point,p,native);CHECK(!residual.accepted&&residual.token==0);auto spec=Config();spec->spec_eog_stop=true;s::SamplerState preserved(Sampling(spec));preserved.RecordSpecEogStop(3);CHECK(Count(preserved,"spec_eog_stop",0));}

void DifferentialCandidates(){
 // Complete full-row oracle vs the actual top64 transport, never subset-softmax
 // masquerading as p. Covers masks, raw conservative gate, bias, min_keep/ties.
 unsigned fast=0,fallback=0;std::uint64_t rng=81;
 for(int case_no=0;case_no<120;++case_no){
  auto b=Config();auto c=Sampling(b,1);c.top_k=20;c.top_p=.95;c.min_keep=case_no%5;
  b->initial_thinking=case_no%3==0;b->think_eog_guard=case_no%4!=0;
  b->eog_boundary=case_no%2==0;b->eog_min_prob=case_no%3==1?.2:0;
  b->marker_ids={7,23,61};b->marker_bias=case_no%5==0?-2:0;
  b->logit_bias_enabled=true;b->logit_bias={{9,case_no%7==0?1.5F:-1.0F},{75,-.7F}};
  std::vector<float> full(100);for(auto& value:full)value=static_cast<float>(s::Uniform(&rng)*18-9);
  if(case_no%10==0)for(unsigned i=0;i<100;++i)full[i]=static_cast<float>(i%3); // ties at shortlist boundary
  std::vector<s::TokenId> ids(100);for(unsigned i=0;i<100;++i)ids[i]=i;
  std::ranges::sort(ids,[&](auto a,auto z){return full[a]==full[z]?a<z:full[a]>full[z];});ids.resize(64);
  std::vector<float> values;for(auto id:ids)values.push_back(full[id]);
  s::SamplerState sampler(c);if(!b->initial_thinking)sampler.Accept(2);
  auto top=sampler.DistributionFromTop(values,ids,100);auto all=sampler.Distribution(full);
  if(top){++fast;for(unsigned i=0;i<100;++i)CHECK(std::abs(top->probability(i)-all.probability(i))<1e-11);}else ++fallback;
 }
 CHECK(fast>30);CHECK(fallback>0);
 auto b=Config();b->eog_min_prob=.2;b->eog_min_prob_exact=true;auto c=Sampling(b,1);c.top_k=2;s::SamplerState exact(c);
 std::vector<float> values={3,2,1};std::vector<s::TokenId> ids={2,0,1};CHECK(!exact.DistributionFromTop(values,ids,10));
 // O/tests/test-eog-min-prob.cpp:70 intentional sparse upper false negative.
 auto upper=Config();upper->eog_ids={0};upper->eog_min_prob=.2;c=Sampling(upper,1);c.top_k=2;
 const std::vector<float> raw={std::log(.19F),std::log(.21F),std::log(.60F)};
 s::SamplerState conservative(c);CHECK(conservative.Distribution(raw).probability(0)==0); // raw top-k excludes EOG here
 // Raise top-k to include EOG while retaining a missing tail: upper passes,
 // exact full-vocabulary gate fails (different declared policies).
 const std::vector<float> tail={std::log(.19F),std::log(.105F),std::log(.60F),std::log(.105F)};
 CHECK(conservative.Distribution(tail).probability(0)>0);upper->eog_min_prob_exact=true;
 s::SamplerState complete(c);CHECK(complete.Distribution(tail).probability(0)==0);
 // Forced token represented in candidate IDs remains a valid singleton.
 auto forced=Config();forced->initial_thinking=true;forced->reasoning_budget_enabled=true;forced->reasoning_budget=0;c=Sampling(forced,1);c.top_k=2;
 s::SamplerState singleton(c);std::vector<float> known={9,8,7,6};std::vector<s::TokenId> known_ids={3,2,11,5};auto target=singleton.DistributionFromTop(known,known_ids,16);CHECK(target&&target->probability(11)==1);
}
// The pre-bound raw_probability arithmetic (sorted denominator), kept as the oracle.
double ReferenceRawProbability(const std::vector<float>& row,s::TokenId token,int top_k,bool exact){
 if(token>=row.size()||!std::isfinite(row[token]))return 0.0;
 std::vector<float> d(row.begin(),row.end());std::erase_if(d,[](float v){return !std::isfinite(v);});
 if(!exact&&top_k>0&&d.size()>static_cast<std::size_t>(top_k)){
  std::partial_sort(d.begin(),d.begin()+top_k,d.end(),std::greater<>{});d.resize(top_k);
 }else std::ranges::sort(d,std::greater<>{});
 const double maximum=d.front();double sum=0;for(auto v:d)sum+=std::exp(static_cast<double>(v)-maximum);
 if(sum<=0)return 0.0;
 return std::min(1.0,std::exp(static_cast<double>(row[token])-maximum)/sum);
}
void EogBoundDecisionIdentity(){
 // The max-logit bound must reproduce the sorted denominator's decisions:
 // min-prob masks and the defer flag, top_k 0 / 20, exact off / on, on random
 // rows and at the float steps where the EOG probability crosses the threshold.
 std::uint64_t rng=4242;unsigned bound_settled=0,denominator_used=0,checked=0;
 for(int case_no=0;case_no<480;++case_no){
  const int top_k=case_no%2==0?0:20;const bool exact=(case_no/2)%2==1;const bool defer=(case_no/4)%2==1;
  const double threshold=case_no%3==0?(defer?.9:.2):(defer?.5+s::Uniform(&rng)*.49:.01+s::Uniform(&rng)*.94);
  std::vector<float> row(256);
  for(auto& v:row)v=static_cast<float>(-30+s::Uniform(&rng)*20);  // tail below -10
  const double spread=.2+s::Uniform(&rng)*(defer?8:4);
  row[2]=0;for(unsigned i=3;i<12;++i)row[i]=static_cast<float>(-s::Uniform(&rng)*spread);
  if(case_no%5==0)row[4]=row[2];  // tie at the maximum
  if(case_no%7==0){row[20]=-INFINITY;row[21]=-INFINITY;}
  const bool second_eog=case_no%6==0;
  row[1]=second_eog?static_cast<float>(-s::Uniform(&rng)*3):-INFINITY;
  const auto reference_p=[&](float eog){auto r=row;r[0]=eog;return ReferenceRawProbability(r,0,top_k,exact);};
  // Bisect the EOG logit where the reference crosses the threshold.
  double lo=-9,hi=20;
  for(int i=0;i<80;++i){const double mid=(lo+hi)/2;(reference_p(static_cast<float>(mid))>=threshold?hi:lo)=mid;}
  const float lo_f=static_cast<float>(lo),hi_f=static_cast<float>(hi);
  // Crossing steps use the denominator; the steps at log(threshold) are the
  // closest cases the bound settles (the row maximum is 0 there).
  const float bound_edge=static_cast<float>(std::log(threshold));
  const std::vector<float> eogs={std::nextafter(lo_f,-INFINITY),lo_f,hi_f,std::nextafter(hi_f,INFINITY),
   std::nextafter(bound_edge,-INFINITY),bound_edge,std::nextafter(bound_edge,INFINITY),
   static_cast<float>(-9+s::Uniform(&rng)*10),static_cast<float>(-8+s::Uniform(&rng)*4)};
  for(const float eog:eogs){
   row[0]=eog;
   auto b=Config();b->eog_min_prob_exact=exact;
   if(defer){b->eog_boundary=true;b->eog_defer_p=threshold;}else b->eog_min_prob=threshold;
   auto c=Sampling(b,1);c.top_k=top_k;c.top_p=1;
   s::SamplerState sampler(c);if(defer)sampler.Accept(2);  // "word": not at a boundary
   const auto d=sampler.Distribution(row);
   double finite_max=-INFINITY;for(auto v:row)if(std::isfinite(v))finite_max=std::max(finite_max,static_cast<double>(v));
   bool expect_defer=false;
   for(s::TokenId token:{s::TokenId{0},s::TokenId{1}}){
    if(!std::isfinite(row[token]))continue;
    const double p=ReferenceRawProbability(row,token,top_k,exact);
    if(std::exp(static_cast<double>(row[token])-finite_max)<threshold)++bound_settled;else ++denominator_used;
    if(defer)expect_defer=expect_defer||p>=threshold;
    else CHECK((d.probability(token)==0)==(p<threshold));  // EOG stays in the top 20 when unmasked
    ++checked;
   }
   if(defer)CHECK(d.behaviour()&&d.behaviour()->defer_stop==expect_defer);
  }
 }
 CHECK(bound_settled>500);CHECK(denominator_used>1000);CHECK(checked>4000);
}
void GrammarPhaseAndReset(){
 // Thinking bytes and a literal split tool-call trigger cannot activate the
 // lazy grammar until the close; tentative copy rollback keeps it inactive.
 auto b=Config();b->initial_thinking=true;b->think_eog_guard=true;b->reasoning_budget_enabled=true;b->reasoning_budget=1;
 auto pieces=std::make_shared<std::vector<std::string>>(*b->token_pieces);(*pieces)[4]="<tool_";(*pieces)[5]="call>";(*pieces)[6]="{}";(*pieces)[7]="\n";(*pieces)[8]="<function=f>";b->token_pieces=pieces;
 auto vocabulary=std::make_shared<s::ConstraintVocabulary>(16,[&](std::uint32_t id){return s::ConstraintVocabulary::Piece{(*pieces)[id],id<2};});
 auto grammar=s::JsonConstraint::WithTools(nullptr,{{"f",s::JsonConstraint::Object()}},false,false,s::JsonConstraint::ToolFormat::kQwen);
 auto constraint=std::make_shared<s::TokenConstraint>();constraint->grammar=grammar;constraint->vocabulary=vocabulary;
 auto c=Sampling(b);c.constraint=constraint;s::SamplerState sampler(c);CHECK(!sampler.NeedsConstraintMask());auto tentative=sampler;tentative.Accept(4);tentative.Accept(5);CHECK(!sampler.NeedsConstraintMask());
 std::vector<float> row(16,-10);row[0]=10;row[4]=9;sampler.Accept(4);CHECK(sampler.Sample(row)==11);sampler.Accept(11);CHECK(sampler.Sample(row)==12);sampler.Accept(12);CHECK(!sampler.NeedsConstraintMask());
 sampler.Accept(4);sampler.Accept(5);CHECK(sampler.NeedsConstraintMask());sampler.ResetHistory({});CHECK(!sampler.NeedsConstraintMask());
}
void DrawRollback(){auto b=Config();b->think_eog_guard=true;b->initial_thinking=true;s::SamplerState sampler(Sampling(b));std::vector<float> row={10,9,8};auto saved=sampler.SaveDrawState();CHECK(sampler.Sample(row)==2);CHECK(Count(sampler,"think_eog_guard",1));sampler.RestoreDrawState(saved);CHECK(Count(sampler,"think_eog_guard",0));CHECK(sampler.Sample(row)==2);CHECK(Count(sampler,"think_eog_guard",1));}
void MultiBlockClone(){auto b=Config();b->reasoning_budget_enabled=true;b->reasoning_budget=2;b->think_starts={{8,9}};b->think_ends={{13,14}};s::SamplerState sampler(Sampling(b));std::vector<float> row(16,-10);row[2]=10;sampler.Accept(8);sampler.Accept(9);sampler.Accept(2);auto copy=sampler;sampler.Accept(2);CHECK(sampler.Sample(row)==11);CHECK(copy.Sample(row)==2);sampler.Accept(11);auto forcing_copy=sampler;CHECK(forcing_copy.Sample(row)==12);sampler.Accept(12);sampler.Accept(10);sampler.Accept(13);sampler.Accept(14);CHECK(sampler.Sample(row)==2);sampler.Accept(10);sampler.Accept(2);sampler.Accept(2);CHECK(sampler.Sample(row)==11);}
void RedundantInterventions(){auto b=Config();b->initial_thinking=true;b->think_eog_guard=true;b->logit_bias_enabled=true;b->logit_bias={{2,20}};s::SamplerState sampler(Sampling(b));std::vector<float> row={10,9,8};CHECK(sampler.Sample(row)==2);CHECK(Count(sampler,"logit_bias",0));CHECK(Count(sampler,"think_eog_guard",0));}
void CandidateGlobalTies(){auto b=Config();b->logit_bias_enabled=true;b->logit_bias={{7,1}};auto c=Sampling(b,1);c.top_k=1;s::SamplerState sampler(c);std::vector<float> full(100);for(unsigned i=0;i<100;++i)full[i]=-100-static_cast<float>(i);full[3]=4;full[7]=3;full[12]=2;std::vector<s::TokenId> ids={7,3,12};for(unsigned i=0;ids.size()<64;++i)if(i!=3&&i!=7&&i!=12)ids.push_back(i);std::vector<float> values;for(auto id:ids)values.push_back(full[id]);auto top=sampler.DistributionFromTop(values,ids,100);CHECK(top.has_value());auto dense=sampler.Distribution(full);CHECK(top->probability(3)==1);for(unsigned i=0;i<100;++i)CHECK(top->probability(i)==dense.probability(i));}
void Marker133Ids(){
  // This is a transport/storage proof, not a model tokenizer-resolution proof.
  auto b=Config();b->initial_thinking=true;b->marker_bias=-2;
  for(s::TokenId id=120;id<=252;++id)b->marker_ids.push_back(id);
  CHECK(b->marker_ids.size()==133);
  s::SamplerState sampler(Sampling(b));std::vector<float> row(256,-10);
  row[252]=5;row[253]=4;
  CHECK(sampler.Sample(row)==253);CHECK(Count(sampler,"marker_bias",1));
  row[252]=-10;
  CHECK(sampler.Sample(row)==253);CHECK(Count(sampler,"marker_bias",1));
}
void ModelSuppression(){auto b=Config();b->model_suppress_tokens=true;b->suppress_ids={2};b->logit_bias_enabled=true;b->logit_bias={{2,100}};s::SamplerState sampler(Sampling(b));std::vector<float> row={-10,-10,5,4};CHECK(sampler.Sample(row)==3);CHECK(Count(sampler,"model_suppress_tokens",1));row[2]=-1000;CHECK(sampler.Sample(row)==3);CHECK(Count(sampler,"model_suppress_tokens",1));}
void PresetRealChoice(){
 auto b=Config();b->production_sampling=true;b->native_temperature=0;b->native_top_k=0;b->native_presence_penalty=1.5;
 auto c=Sampling(b);s::SamplerState sampler(c);std::vector<float> row={-10,-10,3,2};sampler.Accept(2);CHECK(sampler.Sample(row)==2);CHECK(Count(sampler,"production_sampling",1));
 auto same=Config();same->production_sampling=true;same->native_temperature=0;auto c2=Sampling(same);s::SamplerState unchanged(c2);CHECK(unchanged.Sample(row)==2);CHECK(Count(unchanged,"production_sampling",0));
}
void LongThinkTransition() {
  auto b = Config();
  b->initial_thinking = true;
  b->loop_breaker = true;
  b->loop_transition = 2;
  b->loop_eog_close = true;
  b->think_eog_guard = true;
  b->loop_prefix = {4, 5};
  b->budget_forced = {6, 11, 12};
  s::SamplerState sampler(Sampling(b));
  for (int i = 0; i < 704; ++i)
    sampler.Accept(100 + i % 44);
  auto state = sampler.SaveDrawState();
  CHECK(state.behaviour->loop.nudges == 1 && !state.behaviour->forcing());
  for (int i = 704; i < 720; ++i)
    sampler.Accept(100 + i % 44);
  CHECK(sampler.SaveDrawState().behaviour->force_reason == "loop_transition");
  std::vector<float> row(256, -10);
  row[0] = 10;
  row[2] = 9;
  for (auto token : {4u, 5u}) {
    CHECK(sampler.Sample(row) == token);
    sampler.Accept(token);
  }
  state = sampler.SaveDrawState();
  CHECK(state.behaviour->prefix_used && state.behaviour->loop.count == 0);
  auto backstop = sampler;
  CHECK(!sampler.CanSelectArgmax(0));
  CHECK(sampler.Distribution(row).probability(11) == 1);
  for (auto token : {11u, 12u}) {
    CHECK(sampler.Sample(row) == token);
    sampler.Accept(token);
  }
  CHECK(!sampler.SaveDrawState().behaviour->thinking);
  CHECK(sampler.Sample(row) == 0);
  for (int i = 0; i < 704; ++i)
    backstop.Accept(100 + i % 44);
  CHECK(backstop.SaveDrawState().behaviour->force_reason == "loop_breaker");
  // An arbitrary backstop must include the budget message before the close.
  for (auto token : {6u, 11u, 12u}) {
    CHECK(backstop.Sample(row) == token);
    backstop.Accept(token);
  }
  CHECK(!backstop.SaveDrawState().behaviour->thinking);
  auto tools = std::make_shared<s::BehaviourConfig>(*b);
  tools->loop_exempt = true;
  s::SamplerState exempt(Sampling(tools));
  for (int i = 0; i < 4096; ++i)
    exempt.Accept(100 + i % 44);
  CHECK(exempt.SaveDrawState().behaviour->loop.detections == 0);
  CHECK(!exempt.SaveDrawState().behaviour->prefix_used);
  CHECK(exempt.Sample(row) == 2);  // Guard remains active; no loop redirect.
}

void GuardEscalation() {
  auto b = Config();
  b->initial_thinking = true;
  b->think_eog_guard = true;
  b->loop_eog_close = true;
  CHECK(b->think_guard_escalate == 16);
  s::SamplerState sampler(Sampling(b));
  std::vector<float> row(16, -10);
  row[0] = 10;
  row[2] = 9;
  for (int i = 1; i <= 16; ++i) {
    // A speculative worker may accept before publishing its draw journal.
    auto working = sampler;
    CHECK(working.Sample(row) == 2);
    CHECK(working.SaveDrawState().behaviour->guard_blocked_attempts ==
          static_cast<unsigned>(i - 1));
    working.Accept(2);
    sampler.CopyDrawStateFrom(working);
    CHECK(sampler.SaveDrawState().behaviour->guard_blocked_attempts ==
          static_cast<unsigned>(i - 1));
    sampler.Accept(2);
    auto state = sampler.SaveDrawState();
    CHECK(state.behaviour->guard_blocked_attempts == static_cast<unsigned>(i));
    CHECK(state.behaviour->guard_escalated == (i == 16));
    CHECK(Count(sampler, "guard_escalation", i == 16 ? 1 : 0));
    if (i == 15)
      CHECK(sampler.Distribution(row).probability(0) == 0);
  }
  CHECK(!sampler.CanSelectArgmax(0));
  CHECK(sampler.Distribution(row).probability(0) == 0);
  CHECK(sampler.Distribution(row).probability(11) == 1);
  // Complete candidate transport agrees with the full probability redirect.
  auto candidate_config = Sampling(b, 1);
  candidate_config.top_k = 1;
  s::SamplerState candidates(candidate_config);
  for (int i = 0; i < 16; ++i) {
    CHECK(candidates.Sample(row) == 2);
    candidates.Accept(2);
  }
  std::vector<s::TokenId> ids;
  for (unsigned i = 0; i < 16; ++i)
    ids.push_back(i);
  auto top = candidates.DistributionFromTop(row, ids, 16);
  CHECK(top && top->probability(0) == 0 && top->probability(11) == 1);
  CHECK(candidates.SampleFromTop(row, ids, 16) == 11);
  for (auto token : {11u, 12u}) {
    CHECK(sampler.Sample(row) == token);
    sampler.Accept(token);
  }
  auto state = sampler.SaveDrawState();
  CHECK(!state.behaviour->thinking && !state.behaviour->guard_escalated);
  CHECK(state.behaviour->guard_blocked_attempts == 0);
  sampler.Accept(10);
  CHECK(sampler.Sample(row) == 2);
  sampler.Accept(2);
  state = sampler.SaveDrawState();
  CHECK(state.behaviour->thinking && !state.behaviour->guard_escalated);
  CHECK(state.behaviour->guard_blocked_attempts == 1);
  CHECK(Count(sampler, "guard_escalation",
              1));  // One per request, not one per span.
  bool reason = false;
  for (const auto& line : sampler.BehaviourSummary())
    if (line ==
        "BEHAVIOUR_ENGAGE name=guard_escalation taken=1 fallback=0 "
        "reason=blocked_attempts_reached_k scope=request")
      reason = true;
  CHECK(reason);
  sampler.ResetHistory({});
  CHECK(Count(sampler, "guard_escalation", 0));
}

void GuardAcceptedOnlyAndOff() {
  auto b = Config();
  b->initial_thinking = true;
  b->think_eog_guard = true;
  b->loop_eog_close = true;
  b->think_guard_escalate = 1;
  s::SamplerState sampler(Sampling(b));
  std::vector<float> row(16, -10);
  row[0] = 10;
  row[2] = 9;
  const auto saved = sampler.SaveDrawState();
  q::MtpProposal proposal;
  proposal.size = 1;
  proposal.ids[0] = 0;
  proposal.probabilities[0] = 1;
  proposal.token = 0;
  proposal.probability = 1;
  auto draft = sampler;
  auto rejected = q::VerifyMtpProposal(row, proposal, draft);
  CHECK(!rejected.accepted && rejected.token == 2);
  CHECK(draft.SaveDrawState().behaviour->guard_blocked_attempts == 0);
  CHECK(!draft.SaveDrawState().behaviour->guard_escalated);
  // Discard the draft position: it must not leak into the committed state.
  sampler.Accept(2);
  CHECK(sampler.SaveDrawState().behaviour->guard_blocked_attempts == 0);
  sampler.RestoreDrawState(saved);
  CHECK(sampler.Sample(row) == 2);
  CHECK(!sampler.SaveDrawState().behaviour->guard_escalated);
  sampler.RestoreDrawState(saved);
  sampler.Accept(2);
  CHECK(!sampler.SaveDrawState().behaviour->guard_escalated);
  // Keeping the target residual counts once when that token is accepted.
  draft.Accept(rejected.token);
  CHECK(draft.SaveDrawState().behaviour->guard_escalated);
  auto off = std::make_shared<s::BehaviourConfig>(*b);
  off->think_guard_escalate = 0;
  s::SamplerState disabled(Sampling(off));
  for (int i = 0; i < 64; ++i) {
    CHECK(disabled.Sample(row) == 2);
    disabled.Accept(2);
  }
  CHECK(disabled.SaveDrawState().behaviour->guard_blocked_attempts == 64);
  CHECK(!disabled.SaveDrawState().behaviour->guard_escalated);
  CHECK(Count(disabled, "guard_escalation", 0));
}

int main(){try{
  const auto run=[](const char* name,auto test){std::cout<<"RUN "<<name<<std::endl;test();};
  run("BoundaryPort",BoundaryPort);run("LoopPort",LoopPort);
  run("LoopNearMissRecurrence",LoopNearMissRecurrence);run("UnsetIdentity",UnsetIdentity);
  run("GuardBudgetBias",GuardBudgetBias);run("MinProbDefer",MinProbDefer);
  run("CandidateProof",CandidateProof);run("CloseReplay",CloseReplay);
  run("SpecExactness",SpecExactness);run("DifferentialCandidates",DifferentialCandidates);
  run("GrammarPhaseAndReset",GrammarPhaseAndReset);run("PresetRealChoice",PresetRealChoice);
  run("ModelSuppression",ModelSuppression);run("DrawRollback",DrawRollback);
  run("MultiBlockClone",MultiBlockClone);run("RedundantInterventions",RedundantInterventions);
  run("CandidateGlobalTies",CandidateGlobalTies);run("Marker133Ids",Marker133Ids);
  run("LongThinkTransition",LongThinkTransition);run("GuardEscalation",GuardEscalation);
  run("GuardAcceptedOnlyAndOff",GuardAcceptedOnlyAndOff);
  run("EogBoundDecisionIdentity",EogBoundDecisionIdentity);
  std::cout<<"behaviour_test: 22 groups PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
