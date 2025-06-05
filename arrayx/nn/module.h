#pragma once

#include "jit.h"

namespace ax::nn {
    class Module {
    protected:
        Array input;
        static Jit jit;

    public:
        Module() = default;

        virtual ~Module() = default;

        Module(const Module &) = delete;

        Module &operator=(const Module &) = delete;

        virtual Array forward(const Array &input) = 0;

        Array operator()(const Array &input) {
            const JitKey key(input);
            Array output = jit(input, [this](const Array &input) {
                this->input = input.detach();
                return forward(this->input);
            });
            output.eval();
            return output;
        }
    };

    inline Jit Module::jit;
} // namespace ax::nn