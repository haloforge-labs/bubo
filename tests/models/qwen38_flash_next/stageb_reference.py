"""Independent FP64 CPU oracle for the routed branch, excluding shared expert.

Inputs are ten (already projected) expert vectors and their original normalized
weights. Equal stored weights retain the supplied router slot rank: division
can round distinct pre-normalization probabilities to the same FP32 value.
This is also the source of expected values for stageb_ops_test.
"""
import math

KS = (2, 3, 4, 6, 10)


def reference(experts, weights, inject, parts, ids=None):
    if (len(experts) != 10 or len(weights) != 10 or not parts or
            len(inject) != 4 * parts or not experts[0] or
            any(len(e) != len(experts[0]) for e in experts) or
            any(not math.isfinite(x) for e in experts for x in e) or
            any(not math.isfinite(x) or x < 0 for x in weights) or
            abs(sum(weights) - 1) > 1e-5 or
            any(not math.isfinite(x) for x in inject)):
        raise ValueError('invalid stage B oracle inputs')
    ids = list(range(10)) if ids is None else ids
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError('invalid expert ids')
    order = sorted(range(10), key=lambda s: -weights[s])
    full = [math.fsum(weights[s] * experts[s][i] for s in order)
            for i in range(len(experts[0]))]
    ff = math.fsum(x*x for x in full)
    gates = []
    for s in range(4):
        x = sum(inject[s*parts:(s+1)*parts]) / 4
        # Stable CPU sigmoid, including finite router/inject saturation.
        gates.append(2 / (1 + math.exp(-x)) if x >= 0 else
                     2 * math.exp(x) / (1 + math.exp(x)))
    totals = dict(rows=1, top1_mass=weights[order[0]],
                  top4_mass=sum(weights[s] for s in order[:4]),
                  gate_l2=math.sqrt(sum(g*g for g in gates)), k={})
    for k in KS:
        kept = full if k == 10 else [
            math.fsum(weights[s] * experts[s][i] for s in order[:k])
            for i in range(len(full))]
        totals['k'][str(k)] = dict(rf=math.fsum(x*y for x, y in zip(kept, full)),
                                    rr=math.fsum(x*x for x in kept), ff=ff,
                                    error2=math.fsum((x-y)**2 for x, y in zip(full, kept)),
                                    omitted_mass=0. if k == 10 else 1 - sum(weights[s] for s in order[:k]))
    return dict(totals=totals, gates=gates, gate_l2=totals['gate_l2'],
                gate_max=max(gates), f_norm=math.sqrt(ff))
