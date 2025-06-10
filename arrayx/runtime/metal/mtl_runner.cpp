#include "mtl_runner.h"

namespace ax::runtime::metal {
    void MTLRunner::run_initializer_op(OpPtr op) {
        LazyPtr lazy = op->get_lazy();
        switch (op->get_opcode()) {
        case Opcode::FULL: {
            alloc(lazy);
            std::shared_ptr<FullOp> full_op = std::static_pointer_cast<FullOp>(op);
            run_full_kernel(op, full_op->get_const());
            break;
        }
        case Opcode::ARANGE: {
            alloc(lazy);
            std::shared_ptr<ArangeOp> arange_op = std::static_pointer_cast<ArangeOp>(op);
            run_arange_kernel(op, arange_op->get_start(), arange_op->get_step());
            break;
        }
        default:
            break;
        }
    }

    void MTLRunner::run_unary_op(OpPtr op) {
        std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
        LazyPtr out_lazy = unary_op->get_lazy();
        OpPtr operand = unary_op->get_operand();

        if (unary_op->is_in_place()) {
            alloc(out_lazy, operand->get_lazy());
        } else {
            alloc(out_lazy);
        }

        if (unary_op->get_opcode() == Opcode::COPY) {
            run_copy_kernel(operand, op);
        } else {
            run_unary_kernel(unary_op->get_opname(), operand, op);
        }
    }

    void MTLRunner::run_binary_op(OpPtr op) {
        std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
        LazyPtr out_lazy = binary_op->get_lazy();
        OpPtr lop = binary_op->get_lhs();
        OpPtr rop = binary_op->get_rhs();

        if (binary_op->get_mode() == BinaryMode::ELMWISE) {
            std::shared_ptr<ElmwiseBinaryOp> elmwise_op = std::static_pointer_cast<ElmwiseBinaryOp>(binary_op);
            if (elmwise_op->is_in_place()) {
                alloc(out_lazy, lop->get_lazy());
            } else {
                alloc(out_lazy);
            }
        } else {
            alloc(out_lazy);
        }

        if (binary_op->get_mode() == BinaryMode::MATMUL) {
            run_matmul_kernel(lop, rop, op);
        } else {
            run_binary_kernel(binary_op->get_opname(), lop, rop, op);
        }
    }

    void MTLRunner::run_transform_op(OpPtr op) {
        switch (op->get_opcode()) {
        case Opcode::RESHAPE: {
            std::shared_ptr<ReshapeOp> reshape_op = std::static_pointer_cast<ReshapeOp>(op);
            LazyPtr out_lazy = reshape_op->get_lazy();
            OpPtr operand = reshape_op->get_operand();
            LazyPtr in_lazy = operand->get_lazy();
            if (!in_lazy->copy_when_reshape(reshape_op->get_view())) {
                alloc(out_lazy, in_lazy);
            } else {
                alloc(out_lazy);
                run_copy_kernel(operand, op);
            }
            break;
        }
        case Opcode::SLICE: {
            run_simple_transform_op<SliceOp>(op);
            break;
        }
        case Opcode::BROADCAST: {
            run_simple_transform_op<BroadcastOp>(op);
            break;
        }
        case Opcode::PERMUTE: {
            run_simple_transform_op<PermuteOp>(op);
            break;
        }
        case Opcode::SQUEEZE: {
            run_simple_transform_op<SqueezeOp>(op);
            break;
        }
        case Opcode::UNSQUEEZE: {
            run_simple_transform_op<UnsqueezeOp>(op);
            break;
        }
        case Opcode::ASTYPE: {
            std::shared_ptr<AstypeOp> as_type_op = std::static_pointer_cast<AstypeOp>(op);
            OpPtr operand = as_type_op->get_operand();
            alloc(as_type_op->get_lazy());
            run_copy_kernel(operand, op);
            break;
        }
        default:
            break;
        }
    }

    void MTLRunner::run_reduce_op(OpPtr op) {
        std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
        LazyPtr lazy = reduce_op->get_lazy();
        OpPtr operand = reduce_op->get_operand();
        alloc(lazy);
        isize default_val = reduce_op->get_default_val();

        if (reduce_op->get_mode() == ReduceMode::VALUE) {
            // Fill up array with default value
            // With arg operations, the array is already filled up with 0s, which are also the default indices
            // Hence, arg operations do not need to fill up array
            run_full_kernel(op, default_val);
        }

        if (reduce_op->get_dims().size() == 0) {
            // Reduce to one item
            run_reduce_all_kernel(reduce_op->get_opname(), operand, op, default_val);
        } else {
            // Reduce multiple dimensions
            run_reduce_col_kernel(reduce_op->get_opname(), operand, op, default_val);
        }
    }
} // namespace ax::runtime::metal