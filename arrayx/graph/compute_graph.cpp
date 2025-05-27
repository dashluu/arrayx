#include "compute_graph.h"

namespace ax::graph
{
    void ComputeGraph::toposort(OpPtr op, std::vector<OpPtr> &order)
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
            order.push_back(op);
            break;
        }
        case Optype::UNARY:
        {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            toposort(operand, order);
            order.push_back(op);
            break;
        }
        case Optype::BINARY:
        {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            toposort(lhs, order);
            toposort(rhs, order);
            order.push_back(op);
            break;
        }
        case Optype::MATMUL:
        {
            std::shared_ptr<MatmulOp> matmul_op = std::static_pointer_cast<MatmulOp>(op);
            OpPtr lhs = matmul_op->get_lhs();
            OpPtr rhs = matmul_op->get_rhs();
            toposort(lhs, order);
            toposort(rhs, order);
            order.push_back(op);
            break;
        }
        case Optype::TRANSFORM:
        {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            toposort(operand, order);
            order.push_back(op);
            break;
        }
        default:
        {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            toposort(operand, order);
            order.push_back(op);
            break;
        }
        }
    }

    void ComputeGraph::forward()
    {
        if (fw_order.empty())
        {
            toposort(output, fw_order);
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
                op->backward();
            }
            // Order the gradient arrays
            for (auto &op : std::views::reverse(fw_order))
            {
                // grad is null when backward is not implemented for op
                if (op->gradroot != nullptr)
                {
                    toposort(op->gradroot, bw_order);
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