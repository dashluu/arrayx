#pragma once

#include "../graph/compute_graph.h"
#include "../device/buffer.h"

namespace ax::runtime
{
	using namespace ax::graph;
	using namespace ax::device;

	class Runner
	{
	protected:
		virtual void run_full_kernel(OpPtr op, isize c) = 0;

		virtual void run_arange_kernel(OpPtr op, isize start, isize step) = 0;

		virtual void run_binary_kernel(const std::string &name, OpPtr lop, OpPtr rop, OpPtr out_op) = 0;

		virtual void run_matmul_kernel(OpPtr lop, OpPtr rop, OpPtr out_op) = 0;

		virtual void run_unary_kernel(const std::string &name, OpPtr in_op, OpPtr out_op) = 0;

		virtual void run_copy_kernel(OpPtr in_op, OpPtr out_op) = 0;

		virtual void run_reduce_all_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, isize default_val) = 0;

		virtual void run_reduce_col_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, isize default_val) = 0;

		virtual void run_initializer_op(OpPtr op) = 0;

		virtual void run_unary_op(OpPtr op) = 0;

		virtual void run_binary_op(OpPtr op) = 0;

		virtual void run_matmul_op(OpPtr op) = 0;

		virtual void run_transform_op(OpPtr op) = 0;

		virtual void run_reduce_op(OpPtr op) = 0;

		virtual void alloc(LazyArrayPtr arr) = 0;

		virtual void alloc(LazyArrayPtr out_arr, LazyArrayPtr in_arr) = 0;

		void run(OpPtr op);

	public:
		Runner() = default;

		Runner(const Runner &) = delete;

		virtual ~Runner() = default;

		Runner &operator=(const Runner &) = delete;

		void forward(std::shared_ptr<ComputeGraph> graph);

		void backward(std::shared_ptr<ComputeGraph> graph);
	};

	using RunnerPtr = std::shared_ptr<Runner>;
}