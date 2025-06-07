#pragma once

#include "module.h"

namespace ax::nn {
    inline Array relu(const Array &x) {
        return x.maximum(0);
    }

    inline Array linear(const Array &x, const Array &weight) {
        isize weight_ndim = weight.get_ndim();
        return x.matmul(weight.transpose(weight_ndim - 2, weight_ndim - 1));
    }

    inline Array linear_with_bias(const Array &x, const Array &weight, const Array &bias) {
        return linear(x, weight) + bias;
    }

    inline Array onehot(const Array &x, isize num_classes = 0) {
        if (x.get_dtype()->get_type() != DtypeType::INT) {
            throw std::invalid_argument("Array " + x.get_id().str() + " must be of type int.");
        }
        if (num_classes <= 0) {
            num_classes = x.max().item() + 1;
        }
        Array arange = Array::arange({num_classes}, 0, 1, &i32);
        return (x.unsqueeze() == arange).astype(&i32);
    }

    inline Array cross_entropy_loss(const Array &x, const Array &y) {
        /*
        x is logits, y is target
        compute cross entropy loss -sum(y * log(softmax(x)))
        softmax(x) = exp(x) / sum(exp(x))
        log(softmax(x)) = x - log(sum(exp(x)))
        loss = -sum(y * (x - log(sum(exp(x)))))
        loss = -sum(y * x) + sum(y * log(sum(exp(x))))
        sum(y) = 1 and log(sum(exp(x))) is a scalar
        loss = -sum(y * x) + log(sum(exp(x)))
        logsumexp trick for numerical stability:
        log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        x: (*, N)
        y: (*, 1) for labels
        max_x: (*, 1)
        exp_x: (*, N)
        sum_exp_x: (*, 1)
        log_sum_exp_x: (*, 1)
        onehot_y: (*, N)
        onehot_y * x: (*, N)
        loss: (*, 1)
        */
        isize x_ndim = x.get_ndim();
        Array max_x = x.max({x_ndim - 1});
        Array exp_x = (x - max_x).exp();
        Array sum_exp_x = exp_x.sum({x_ndim - 1});
        Array log_sum_exp_x = sum_exp_x.log() + max_x;
        Array onehot_y = onehot(y).astype(x.get_dtype());
        Array loss = -(onehot_y * x).sum({x_ndim - 1}) + log_sum_exp_x;
        return loss.mean();
    }
} // namespace ax::nn