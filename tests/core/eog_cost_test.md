# EOG CPU replay and differential check

`eog_cost_test` links the production `gufo_core` sampler and the existing frozen
release sampler fixture in one executable. The fixture is compiled in a separate
namespace; its source is not copied or modified. Both adapters use the server's
`ParseBehaviourOptions` parser. All input logits and token pieces are synthetic.

Install the production behaviour environment before running:

```text
GUFO_THINK_EOG_GUARD=1 GUFO_EOG_BOUNDARY=1 GUFO_EOG_MIN_PROB=0.2
GUFO_EOG_DEFER_P=0.9 GUFO_SPEC_EOG_STOP=1 GUFO_LOOP_BREAKER=1
GUFO_LOOP_BREAKER_EOG_CLOSE=1 GUFO_LOOP_BREAKER_TRANSITION=2
GUFO_THINK_MARKER_BIAS_BY_EFFORT=low:-2,medium:-1,xhigh:0
GUFO_LENIENT=bounded GUFO_LENIENT_LAMBDA=0.5 GUFO_ROUTED_LANEROW=1
```

```text
eog_cost_test --diff 100000
eog_cost_test --bench old 0 ar
eog_cost_test --bench new 0 ar
eog_cost_test --bench fix 0 ar
eog_cost_test --bench old 20 candidate
eog_cost_test --bench old 0 ar 100 focus
```

`ar` uses the headline request's greedy sampling parameters. `candidate` uses
seeded temperature 1, top-k 20, top-p .95 with the engine's 64 candidates.
`fix` directly interleaves old production and new production. Other benchmark
arms interleave no EOG flags with the selected flags in the same implementation.
Each cell uses seven ABBA repeats. The optional duration is milliseconds per
slice; `focus` selects width 4, non-boundary answer, low-ranked EOG. Widths replay
serial selection and acceptance, as the serve scheduler does. Full-row transfers
are represented by two CPU copies of exactly 993280 bytes; HIP transfer and
synchronization latency are not simulated or measured. This is not a GPU or
complete MTP verifier benchmark.

State IDs: 0 thinking, 1 non-boundary answer, 2 boundary answer, 3 deferred stop;
the differential check additionally covers 4 escalated think and 5 forced close.
Rank IDs: 0 highest EOG, 1 EOG inside top-k, 2 low EOG. Flag bits: 1 think guard,
2 boundary, 4 minimum probability, 8 defer, 16 speculative stop. Differential
rows cover all 32 flag sets, widths 1/4/8, greedy and seeded sampling, top-k
0/1/20, ties, nonfinite EOG and top-p cutoff cases, plus permissive tool grammars.
It compares tokens, RNG/pending draw, behaviour journals and counters before and
after acceptance, with periodic bitwise distribution/stage comparisons.

The timing replay resets state before each row to keep each labelled state
constant. Setup/reset/acceptance and copy controls are included in both arms.
Counter logs are one separate deterministic row probe per cell, not the total
number of timed loop iterations. A `candidate_path` fallback is the actual
sampler rejection reason; an unsuccessful inert-distribution proof simply uses
the unchanged general algorithm and does not add a production counter.
