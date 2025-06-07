#pragma once

#include "bind.h"

namespace ax::bind {
    struct PyModule : axnn::Module {
        NB_TRAMPOLINE(axnn::Module, 2);

        axr::Array forward(const axr::ArrayVec &input) override {
            NB_OVERRIDE_PURE(forward, input);
        }

        axr::ArrayVec parameters() override {
            NB_OVERRIDE_PURE(parameters);
        }
    };
} // namespace ax::bind