#pragma once

#include "../array/array.h"

namespace ax::optim {
    using namespace ax::array;

    class Optimizer {
    protected:
        float lr;
        ArrayVec params;
        ArrayVec grads;
        bool initial_step = false;

    public:
        Optimizer(const ArrayVec &params, float lr = 1e-3) : params(params), lr(lr) {}
        virtual ~Optimizer() = default;
        Optimizer(const Optimizer &) = delete;
        Optimizer &operator=(const Optimizer &) = delete;
        virtual void forward() = 0;

        void step() {
            // Initialize gradients and parameters if not already initialized
            if (!initial_step) {
                for (size_t i = 0; i < params.size(); i++) {
                    auto grad = params[i].get_grad();
                    // Check if gradient exists
                    if (!grad.has_value()) {
                        throw std::runtime_error("Array " + params[i].get_id().str() + " has no gradient for Gradient Descent optimizer.");
                    }
                    // Store detached gradient and parameters
                    grads.push_back(grad.value().detach());
                    params[i] = params[i].detach();
                }

                forward();
                initial_step = true;
            }

            // Evaluate all parameters
            for (Array &param : params) {
                param.eval();
            }
        }
    };

    class GradientDescent : public Optimizer {
    public:
        GradientDescent(const ArrayVec &params, float lr = 1e-3) : Optimizer(params, lr) {
        }

        void forward() override {
            for (size_t i = 0; i < params.size(); i++) {
                params[i] -= lr * grads[i];
            }
        }
    };
} // namespace ax::optim