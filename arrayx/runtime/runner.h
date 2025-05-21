#pragma once

#include "../graph/compute_graph.h"

namespace ax::runtime
{
	using namespace ax::graph;

	class Runner
	{
	protected:
		virtual void run_full_kernel(OpPtr op, int c) = 0;

		virtual void run_arange_kernel(OpPtr op, int start, int step) = 0;

		virtual void run_binary_ss_kernel(const std::string &name, OpPtr lop, OpPtr rop, OpPtr out_op) = 0;

		virtual void run_matmul_kernel(OpPtr lop, OpPtr rop, OpPtr out_op) = 0;

		virtual void run_unary_ss_kernel(const std::string &name, OpPtr in_op, OpPtr out_op) = 0;

		virtual void run_copy_kernel(OpPtr in_op, OpPtr out_op) = 0;

		virtual void run_reduce_all_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, int default_val) = 0;

		virtual void run_reduce_col_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, int default_val) = 0;

		virtual void run_initializer_op(OpPtr op) = 0;

		virtual void run_unary_op(OpPtr op) = 0;

		virtual void run_binary_op(OpPtr op) = 0;

		virtual void run_matmul_op(OpPtr op) = 0;

		virtual void run_transform_op(OpPtr op) = 0;

		virtual void run_reduce_op(OpPtr op) = 0;

		virtual void alloc(ArrayPtr arr) = 0;

		virtual void alloc(ArrayPtr arr, std::shared_ptr<Buffer> buff) = 0;

		void run(OpPtr op);

	public:
		void forward(std::shared_ptr<ComputeGraph> graph);

		void backward(std::shared_ptr<ComputeGraph> graph);
	};
}