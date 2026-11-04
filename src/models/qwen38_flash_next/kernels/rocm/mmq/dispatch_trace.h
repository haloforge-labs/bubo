#pragma once
#include <stdint.h>

// Host-only dispatch observations. No model kernel reads these structures.
struct qfn_mmq_dispatch {
    uint32_t token_tile, feature_tile, grid_x, grid_y, grid_z;
    int weight_type;
};
struct qfn_mmq_dispatch_trace {
    uint32_t count, invalid;
    qfn_mmq_dispatch dispatches[4];
};
