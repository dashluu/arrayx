#pragma once

#include "bind.h"

namespace ax::bind {
    struct PyOptimizer : axo::Optimizer {
        NB_TRAMPOLINE(axo::Optimizer, 2);

        void optim_func() override {
            NB_OVERRIDE_PURE(optim_func);
        }

        void step() override {
            NB_OVERRIDE_PURE(step);
        }
    };
} // namespace ax::bind