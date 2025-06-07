#pragma once

#include "jit.h"

namespace ax::nn {
    class Module {
    protected:
        Array input;
        Jit jit_engine;

    public:
        Module() = default;

        virtual ~Module() = default;

        Module(const Module &) = delete;

        Module &operator=(const Module &) = delete;

        virtual Array forward(const Array &input) = 0;

        virtual ArrayVec parameters() = 0;

        Array operator()(const Array &input) { return forward(input); }

        Array jit(const Array &input) {
            Array output = jit_engine(input, [this](const Array &input) {
                this->input = Array::empty_twin(input);
                return forward(this->input);
            });
            this->input.set_lazy(input);
            return output;
        }
    };
} // namespace ax::nn