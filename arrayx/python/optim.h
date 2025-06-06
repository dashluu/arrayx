#pragma once

#include "bind.h"

namespace ax::bind {
    struct PyOptimizer : axo::Optimizer {
        NB_TRAMPOLINE(axo::Optimizer, 1);

        void forward() override {
            NB_OVERRIDE_PURE(forward);
        }
    };
} // namespace ax::bind