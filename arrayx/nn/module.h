#pragma once

#include "jit.h"

namespace ax::nn {
    class Module {
    protected:
        ArrayVec input;
        Jit jit_engine;

    public:
        Module() = default;

        virtual ~Module() = default;

        Module(const Module &) = delete;

        Module &operator=(const Module &) = delete;

        virtual Array forward(const ArrayVec &input) = 0;

        virtual ArrayVec parameters() { return {}; }

        Array operator()(const ArrayVec &input) { return forward(input); }

        Array jit(const ArrayVec &input) {
            Array output = jit_engine(input, [this](const ArrayVec &input) {
                for (auto &arr : input) {
                    this->input.push_back(Array::empty_twin(arr));
                }
                return forward(this->input);
            });
            for (size_t i = 0; i < input.size(); i++) {
                this->input[i].set_lazy(input[i]);
            }
            return output;
        }
    };
} // namespace ax::nn