#pragma once

#include "module.h"

namespace ax::nn
{
	inline Array relu(const Array &x)
	{
		return (x >= 0).astype(&f32) * x;
	}

	inline Array onehot(const Array &x, isize num_classes)
	{
		if (x.get_dtype()->get_type() != DtypeType::INT)
		{
			throw std::invalid_argument("Array " + x.get_id().str() + " must be of type int.");
		}
		if (num_classes <= 0)
		{
			num_classes = x.max().item() + 1;
		}
		auto arange = Array::arange({num_classes}, 0, 1, &i32);
		return (x.unsqueeze() == arange).astype(&i32);
	}

	inline Array cross_entropy_loss(const Array &x, const Array &y, isize num_classes)
	{
		/*
		x is logits, y is target
		compute cross entropy loss
		-sum(y * log(softmax(x)))
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
		auto max_x = x.max({-1});
		auto exp_x = (x - max_x).exp();
		auto sum_exp_x = exp_x.sum({-1});
		auto log_sum_exp_x = sum_exp_x.log() + max_x;
		auto onehot_y = onehot(y, num_classes).astype(&f32);
		auto loss = -(onehot_y * x).sum({-1}) + log_sum_exp_x;
		return loss.mean();
	}
}