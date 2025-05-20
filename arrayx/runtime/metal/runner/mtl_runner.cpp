#include "mtl_runner.h"

namespace ax::runtime::metal
{
	void MTLRunner::run_initializer_op(OpPtr op)
	{
		ArrayPtr arr = op->get_output();
		arr->alloc();
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
			arr->alloc(*(operand->get_output())->get_buff());
		}
		else
		{
			arr->alloc();
		}
		run_unary_ss_kernel(unary_op->get_opcode_str(), operand, op);
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
			arr->alloc(*(lop->get_output())->get_buff());
		}
		else
		{
			arr->alloc();
		}
		run_binary_ss_kernel(binary_op->get_opcode_str(), lop, rop, op);
	}

	void MTLRunner::run_matmul_op(OpPtr op)
	{
		std::shared_ptr<MatmulOp> matmul_op = std::static_pointer_cast<MatmulOp>(op);
		OpPtr lop = matmul_op->get_lhs();
		OpPtr rop = matmul_op->get_rhs();
		matmul_op->get_output()->alloc();
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
				out_arr->alloc(*in_arr->get_buff());
			}
			else
			{
				out_arr->alloc();
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
		case Opcode::AS_TYPE:
		{
			std::shared_ptr<AsTypeOp> as_type_op = std::static_pointer_cast<AsTypeOp>(op);
			OpPtr operand = as_type_op->get_operand();
			op->get_output()->alloc();
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
		arr->alloc();
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

	void MTLRunner::run(OpPtr op)
	{
		switch (op->get_optype())
		{
		case Optype::INITIALIZER:
		{
			run_initializer_op(op);
			break;
		}
		case Optype::UNARY:
		{
			run_unary_op(op);
			break;
		}
		case Optype::BINARY:
		{
			run_binary_op(op);
			break;
		}
		case Optype::MATMUL:
		{
			run_matmul_op(op);
			break;
		}
		case Optype::TRANSFORM:
		{
			run_transform_op(op);
			break;
		}
		default:
		{
			run_reduce_op(op);
			break;
		}
		}
	}

	void MTLRunner::forward(std::shared_ptr<ComputeGraph> graph)
	{
		for (auto iter = graph->cbegin(); iter != graph->cend(); ++iter)
		{
			run(*iter);
		}
	}

	void MTLRunner::backward(std::shared_ptr<ComputeGraph> graph)
	{
		for (auto iter = graph->crbegin(); iter != graph->crend(); ++iter)
		{
			run(*iter);
		}
	}
}