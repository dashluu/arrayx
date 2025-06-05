#pragma once

#include "../array/array.h"

namespace ax::optim {
    using namespace ax::array;

    class Optimizer {
      protected:
        float lr;
        std::vector<Array> params;

      public:
        Optimizer(std::vector<Array> params, float lr = 1e-3) : lr(lr) {
            for (auto &p : params) {
                auto param = p.detach();
                this->params.push_back(param);
            }
        }

        virtual ~Optimizer() = default;
        Optimizer(const Optimizer &) = delete;
        Optimizer &operator=(const Optimizer &) = delete;
        virtual void optim_func() = 0;
        virtual void step() = 0;
    };

    class GradientDescent : public Optimizer {
      public:
        GradientDescent(std::vector<Array> params, float lr = 1e-3) : Optimizer(params, lr) {
            optim_func();
        }

        void optim_func() override {
            for (size_t i = 0; i < params.size(); i++) {
                std::optional<Array> grad = params[i].get_grad();
                if (!grad.has_value()) {
                    throw std::runtime_error("Array " + params[i].get_id().str() + " has no gradient for Gradient Descent optimizer.");
                }
                params[i] -= lr * grad.value();
            }
        }

        void step() override {
            for (auto &param : params) {
                param.eval();
            }
        }
    };
} // namespace ax::optim