#include "mtl_runner.h"

namespace ax::runtime::metal
{
	void MTLRunner::run_initializer_op(OpPtr op)
	{
		ArrayPtr arr = op->get_output();
		alloc(arr);
		switch (op->get_opcode())
		{
		case Opcode::FULL:
		{
			std::shared_ptr<FullOp> full_op = std::static_pointer_cast<FullOp>(op);
			run_full_kernel(op, full_op->get_const());
			break;
		}
		case Opcode::ARANGE:
		{
			std::shared_ptr<ArangeOp> arange_op = std::static_pointer_cast<ArangeOp>(op);
			run_arange_kernel(op, arange_op->get_start(), arange_op->get_step());
			break;
		}
		default:
			break;
		}
	}

	void MTLRunner::run_unary_op(OpPtr op)
	{
		std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
		ArrayPtr arr = unary_op->get_output();
		OpPtr operand = unary_op->get_operand();
		if (unary_op->is_in_place())
		{
			alloc(arr, operand->get_output()->get_buff());
		}
		else
		{
			alloc(arr);
		}
		if (unary_op->get_opcode() == Opcode::IDENTITY)
		{
			run_copy_kernel(operand, op);
		}
		else
		{
			run_unary_ss_kernel(unary_op->get_opcode_str(), operand, op);
		}
	}

	void MTLRunner::run_binary_op(OpPtr op)
	{
		std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
		ArrayPtr arr = binary_op->get_output();
		OpPtr lop = binary_op->get_lhs();
		OpPtr rop = binary_op->get_rhs();
		if (binary_op->is_in_place())
		{
			// Share memory with lhs
			alloc(arr, lop->get_output()->get_buff());
		}
		else
		{
			alloc(arr);
		}
		run_binary_ss_kernel(binary_op->get_opcode_str(), lop, rop, op);
	}

	void MTLRunner::run_matmul_op(OpPtr op)
	{
		std::shared_ptr<MatmulOp> matmul_op = std::static_pointer_cast<MatmulOp>(op);
		OpPtr lop = matmul_op->get_lhs();
		OpPtr rop = matmul_op->get_rhs();
		alloc(matmul_op->get_output());
		run_matmul_kernel(lop, rop, op);
	}

	void MTLRunner::run_transform_op(OpPtr op)
	{
		switch (op->get_opcode())
		{
		case Opcode::RESHAPE:
		{
			std::shared_ptr<ReshapeOp> reshape_op = std::static_pointer_cast<ReshapeOp>(op);
			ArrayPtr out_arr = reshape_op->get_output();
			OpPtr operand = reshape_op->get_operand();
			ArrayPtr in_arr = operand->get_output();
			if (!in_arr->copy_when_reshape(reshape_op->get_view()))
			{
				alloc(out_arr, in_arr->get_buff());
			}
			else
			{
				alloc(out_arr);
				run_copy_kernel(operand, op);
			}
			break;
		}
		case Opcode::SLICE:
		{
			run_simple_transform_op<SliceOp>(op);
			break;
		}
		case Opcode::BROADCAST:
		{
			run_simple_transform_op<BroadcastOp>(op);
			break;
		}
		case Opcode::PERMUTE:
		{
			run_simple_transform_op<PermuteOp>(op);
			break;
		}
		case Opcode::SQUEEZE:
		{
			run_simple_transform_op<SqueezeOp>(op);
			break;
		}
		case Opcode::UNSQUEEZE:
		{
			run_simple_transform_op<UnsqueezeOp>(op);
			break;
		}
		case Opcode::ASTYPE:
		{
			std::shared_ptr<AstypeOp> as_type_op = std::static_pointer_cast<AstypeOp>(op);
			OpPtr operand = as_type_op->get_operand();
			alloc(as_type_op->get_output());
			run_copy_kernel(operand, op);
			break;
		}
		default:
			break;
		}
	}

	void MTLRunner::run_reduce_op(OpPtr op)
	{
		std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
		ArrayPtr arr = reduce_op->get_output();
		OpPtr operand = reduce_op->get_operand();
		alloc(arr);
		int default_val = reduce_op->get_default_val();
		if (reduce_op->get_mode() == ReduceMode::VALUE)
		{
			// Fill up array with default value
			// With arg operations, the array is already filled up with 0s, which are also the default indices
			// Hence, arg operations do not need to fill up array
			run_full_kernel(op, default_val);
		}
		if (reduce_op->get_dims().size() == 0)
		{
			// Reduce to one item
			run_reduce_all_kernel(reduce_op->get_opcode_str(), operand, op, default_val);
		}
		else
		{
			// Reduce multiple dimensions
			run_reduce_col_kernel(reduce_op->get_opcode_str(), operand, op, default_val);
		}
	}

	void MTLRunner::alloc(ArrayPtr arr)
	{
		arr->buff = std::make_shared<Buffer>(ctx->get_allocator(), arr->get_nbytes());
	}

	void MTLRunner::alloc(ArrayPtr arr, std::shared_ptr<Buffer> buff)
	{
		// Simply assigning buff prevents the memory from being freed properly
		// Reason: shared buffer -> buffer goes out of scope -> Memory is freed twice
		arr->buff = std::make_shared<Buffer>(*buff);
	}
}