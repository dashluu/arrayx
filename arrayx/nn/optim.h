#pragma once

#include "../array/array.h"

namespace ax::optim
{
	using namespace ax::array;

	class Optimizer
	{
	protected:
		float lr;
		std::vector<Array> params;
		virtual void optim_func() = 0;
		virtual void step() = 0;

	public:
		Optimizer(std::vector<Array> params, float lr) : lr(lr)
		{
			for (auto &p : params)
			{
				auto param = p.detach();
				this->params.push_back(param);
			}
			optim_func();
		}
	};

	class GradientDescent : public Optimizer
	{

	private:
		void optim_func() override
		{
			for (size_t i = 0; i < params.size(); i++)
			{
				std::optional<Array> grad = params[i].get_grad();
				if (!grad.has_value())
				{
					throw std::runtime_error("Array " + params[i].get_id().str() + " has no gradient for Gradient Descent optimizer.");
				}
				params[i] -= lr * grad.value();
			}
		}

		void step() override
		{
			for (auto &param : params)
			{
				param.eval();
			}
		}

	public:
		GradientDescent(std::vector<Array> params, float lr = 1e-3) : Optimizer(params, lr) {}
	};
}