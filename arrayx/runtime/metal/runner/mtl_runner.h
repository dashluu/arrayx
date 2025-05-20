#pragma once

#include "../../../graph/compute_graph.h"
#include "mtl_context.h"
#include "mtl_command_encoder.h"

namespace ax::runtime::metal
{
	using namespace ax::graph;

	class MTLRunner
	{
	private:
		std::shared_ptr<MTLContext> ctx;

		void run_full_kernel(OpPtr op, int c);

		void run_arange_kernel(OpPtr op, int start, int step);

		void run_binary_ss_kernel(const std::string &name, OpPtr lop, OpPtr rop, OpPtr out_op);

		void run_matmul_kernel(OpPtr lop, OpPtr rop, OpPtr out_op);

		void run_unary_ss_kernel(const std::string &name, OpPtr in_op, OpPtr out_op);

		void run_copy_kernel(OpPtr in_op, OpPtr out_op);

		void run_reduce_all_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, int default_val);

		void run_reduce_col_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, int default_val);

		void run_initializer_op(OpPtr op);

		void run_unary_op(OpPtr op);

		void run_binary_op(OpPtr op);

		void run_matmul_op(OpPtr op);

		void run_transform_op(OpPtr op);

		template <class O>
		void run_simple_transform_op(OpPtr op)
		{
			auto transform_op = std::static_pointer_cast<O>(op);
			OpPtr operand = transform_op->get_operand();
			op->get_output()->alloc(*(operand->get_output())->get_buff());
		}

		void run_reduce_op(OpPtr op);

		void run(OpPtr op);

	public:
		MTLRunner(std::shared_ptr<MTLContext> ctx) : ctx(ctx) {}

		void forward(std::shared_ptr<ComputeGraph> graph);

		void backward(std::shared_ptr<ComputeGraph> graph);
	};
}