#pragma once

#include "ops.h"

namespace ax::graph
{
	class ComputeGraph : public IStr, public std::enable_shared_from_this<ComputeGraph>
	{
	private:
		OpPtr output;
		std::unordered_set<Id> visited;
		std::vector<OpPtr> fw_order;
		std::vector<OpPtr> bw_order;

		void toposort(OpPtr op, std::vector<OpPtr> &order);

	public:
		ComputeGraph(OpPtr output, const std::vector<OpPtr> &input = {}) : output(output)
		{
			for (auto &op : input)
			{
				visited.insert(op->get_output()->get_id());
			}
		}

		ComputeGraph(const ComputeGraph &) = delete;

		ComputeGraph &operator=(const ComputeGraph &) = delete;

		OpPtr get_output() const { return output; }

		void forward();

		void backward();

		const std::string str() const override;

		std::vector<OpPtr>::const_iterator cbegin() const { return fw_order.begin(); }

		std::vector<OpPtr>::const_iterator cend() const { return fw_order.end(); }

		std::vector<OpPtr>::const_reverse_iterator crbegin() const { return bw_order.crbegin(); }

		std::vector<OpPtr>::const_reverse_iterator crend() const { return bw_order.crend(); }
	};
}