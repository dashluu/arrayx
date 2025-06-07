#pragma once

#include "bind.h"

namespace ax::bind {
    axr::Array jit(axnn::Jit &jit_engine, const axr::Array &x, const nb::callable &callable) {
        if (!callable.is_valid()) {
            throw std::invalid_argument("Invalid callable as JIT argument.");
        }
        return jit_engine(x, callable);
    }

    struct PyModule : axnn::Module {
        NB_TRAMPOLINE(axnn::Module, 2);

        axr::Array forward(const axr::Array &input) override {
            NB_OVERRIDE_PURE(forward, input);
        }

        axr::ArrayVec parameters() override {
            NB_OVERRIDE_PURE(parameters);
        }
    };
} // namespace ax::bind