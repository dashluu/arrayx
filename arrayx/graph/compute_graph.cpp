#include "compute_graph.h"

namespace ax::graph
{
    void ComputeGraph::fw_toposort(OpPtr op)
    {
        LazyArrayPtr arr = op->get_lazy();
        if (visited.contains(arr->get_id()))
        {
            return;
        }
        visited.insert(arr->get_id());
        switch (op->get_optype())
        {
        case Optype::INITIALIZER:
        {
            fw_order.push_back(op);
            break;
        }
        case Optype::UNARY:
        {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            fw_toposort(operand);
            fw_order.push_back(op);
            break;
        }
        case Optype::BINARY:
        {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            fw_toposort(lhs);
            fw_toposort(rhs);
            fw_order.push_back(op);
            break;
        }
        case Optype::MATMUL:
        {
            std::shared_ptr<MatmulOp> matmul_op = std::static_pointer_cast<MatmulOp>(op);
            OpPtr lhs = matmul_op->get_lhs();
            OpPtr rhs = matmul_op->get_rhs();
            fw_toposort(lhs);
            fw_toposort(rhs);
            fw_order.push_back(op);
            break;
        }
        case Optype::TRANSFORM:
        {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            fw_toposort(operand);
            fw_order.push_back(op);
            break;
        }
        default:
        {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            fw_toposort(operand);
            fw_order.push_back(op);
            break;
        }
        }
    }

    void ComputeGraph::bw_toposort(OpPtr op)
    {
        LazyArrayPtr arr = op->get_lazy();
        if (visited.contains(arr->get_id()))
        {
            return;
        }
        visited.insert(arr->get_id());
        switch (op->get_optype())
        {
        case Optype::INITIALIZER:
        {
            bw_order.push_back(op);
            break;
        }
        case Optype::UNARY:
        {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            bw_toposort(operand);
            bw_order.push_back(op);
            break;
        }
        case Optype::BINARY:
        {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            bw_toposort(lhs);
            bw_toposort(rhs);
            bw_order.push_back(op);
            break;
        }
        case Optype::MATMUL:
        {
            std::shared_ptr<MatmulOp> matmul_op = std::static_pointer_cast<MatmulOp>(op);
            OpPtr lhs = matmul_op->get_lhs();
            OpPtr rhs = matmul_op->get_rhs();
            bw_toposort(lhs);
            bw_toposort(rhs);
            bw_order.push_back(op);
            break;
        }
        case Optype::TRANSFORM:
        {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            bw_toposort(operand);
            bw_order.push_back(op);
            break;
        }
        default:
        {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            bw_toposort(operand);
            bw_order.push_back(op);
            break;
        }
        }
    }

    void ComputeGraph::forward()
    {
        if (fw_order.empty())
        {
            fw_toposort(output);
        }
    }

    void ComputeGraph::backward()
    {
        if (fw_order.empty())
        {
            throw ComputeGraphNotForwardedException();
        }
        if (bw_order.empty())
        {
            LazyArrayPtr arr = output->get_lazy();
            if (arr->get_numel() > 1)
            {
                throw std::invalid_argument("Array " + arr->get_id().str() + " must be a singleton to do gradient backpropation.");
            }
            // Initializes gradient with 1's
            output->init_grad(false);
            // Initializes the gradient array first without allocating buffers
            for (auto &op : std::views::reverse(fw_order))
            {
                if (op->grad_enabled)
                {
                    op->backward();
                }
            }
            // Order the gradient arrays
            for (auto &op : std::views::reverse(fw_order))
            {
                // grad is null when backward is not implemented for op or that gradient is disabled
                if (op->grad_root != nullptr)
                {
                    bw_toposort(op->grad_root);
                }
            }
        }
    }

    const std::string ComputeGraph::str() const
    {
        if (fw_order.empty())
        {
            throw ComputeGraphNotForwardedException();
        }
        LazyArrayPtr arr;
        std::string s = "Forward:\n";
        for (auto &op : fw_order)
        {
            arr = op->get_lazy();
            s += op->str() + "\n";
        }
        s += "Backward:\n";
        for (auto &op : bw_order)
        {
            arr = op->get_lazy();
            s += op->str() + "\n";
        }
        return s;
    }
}