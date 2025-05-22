#pragma once

#include "../../runner.h"
#include "mtl_command_encoder.h"

namespace ax::runtime::metal
{
	class MTLRunner : public Runner
	{
	protected:
		std::shared_ptr<MTLContext> ctx;

		void run_full_kernel(OpPtr op, int c) override;

		void run_arange_kernel(OpPtr op, int start, int step) override;

		void run_binary_ss_kernel(const std::string &name, OpPtr lop, OpPtr rop, OpPtr out_op) override;

		void run_matmul_kernel(OpPtr lop, OpPtr rop, OpPtr out_op) override;

		void run_unary_ss_kernel(const std::string &name, OpPtr in_op, OpPtr out_op) override;

		void run_copy_kernel(OpPtr in_op, OpPtr out_op) override;

		void run_reduce_all_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, int default_val) override;

		void run_reduce_col_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, int default_val) override;

		void run_initializer_op(OpPtr op) override;

		void run_unary_op(OpPtr op) override;

		void run_binary_op(OpPtr op) override;

		void run_matmul_op(OpPtr op) override;

		void run_transform_op(OpPtr op) override;

		template <class O>
		void run_simple_transform_op(OpPtr op)
		{
			auto transform_op = std::static_pointer_cast<O>(op);
			OpPtr operand = transform_op->get_operand();
			alloc(op->get_output(), operand->get_output()->get_buff());
		}

		void run_reduce_op(OpPtr op) override;

		void alloc(ArrayPtr arr) override;

		void alloc(ArrayPtr arr, std::shared_ptr<Buffer> buff) override;

	public:
		MTLRunner(std::shared_ptr<MTLContext> ctx) : ctx(ctx) {}
	};
}