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

		void fw_toposort(OpPtr op);
		void bw_toposort(OpPtr op);

	public:
		ComputeGraph(OpPtr output, const std::vector<OpPtr> &input = {}) : output(output)
		{
			for (auto &op : input)
			{
				visited.insert(op->get_lazy()->get_id());
			}
		}

		ComputeGraph(const ComputeGraph &) = delete;

		ComputeGraph &operator=(const ComputeGraph &) = delete;

		OpPtr get_output() const { return output; }

		void forward();

		void backward();

		const std::string str() const override;

		std::vector<OpPtr>::const_iterator cbegin() const { return fw_order.cbegin(); }

		std::vector<OpPtr>::const_iterator cend() const { return fw_order.cend(); }

		std::vector<OpPtr>::const_iterator crbegin() const { return bw_order.cbegin(); }

		std::vector<OpPtr>::const_iterator crend() const { return bw_order.cend(); }
	};
}