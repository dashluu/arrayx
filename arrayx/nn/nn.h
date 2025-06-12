#pragma once

#include "../array/array.h"

namespace ax::nn::functional {
    using namespace ax::array;

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

    inline Array onehot(const Array &x, isize num_classes) {
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
        max: (*, 1)
        exp: (*, N)
        sum_exp: (*, 1)
        log_sum_exp: (*, 1)
        target_onehot: (*, N)
        target_onehot * x: (*, N)
        loss: (*, 1)
        */
        isize ndim = x.get_ndim();
        Array max = x.max({ndim - 1});
        Array exp = (x - max).exp();
        Array sum_exp = exp.sum({ndim - 1});
        Array log_sum_exp = sum_exp.log() + max;
        Array target_onehot = onehot(y, 0).astype(x.get_dtype());
        Array loss = -(target_onehot * x).sum({ndim - 1}) + log_sum_exp;
        return loss.mean();
    }
} // namespace ax::nn