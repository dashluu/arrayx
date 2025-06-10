#include "compute_graph.h"

namespace ax::graph {
    void ComputeGraph::fw_toposort(OpPtr op) {
        LazyPtr lazy = op->get_lazy();
        if (visited.contains(lazy->get_id())) {
            return;
        }
        visited.insert(lazy->get_id());
        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            fw_order.push_back(op);
            break;
        }
        case Optype::UNARY: {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            operand->enable_grad(unary_op->is_grad_enabled());
            fw_toposort(operand);
            fw_order.push_back(op);
            break;
        }
        case Optype::BINARY: {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            lhs->enable_grad(binary_op->is_grad_enabled());
            rhs->enable_grad(binary_op->is_grad_enabled());
            fw_toposort(lhs);
            fw_toposort(rhs);
            fw_order.push_back(op);
            break;
        }
        case Optype::TRANSFORM: {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            operand->enable_grad(transform_op->is_grad_enabled());
            fw_toposort(operand);
            fw_order.push_back(op);
            break;
        }
        default: {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            operand->enable_grad(reduce_op->is_grad_enabled());
            fw_toposort(operand);
            fw_order.push_back(op);
            break;
        }
        }
    }

    void ComputeGraph::bw_toposort(OpPtr op) {
        LazyPtr lazy = op->get_lazy();
        if (visited.contains(lazy->get_id())) {
            return;
        }
        visited.insert(lazy->get_id());
        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            bw_order.push_back(op);
            break;
        }
        case Optype::UNARY: {
            std::shared_ptr<UnaryOp> unary_op = std::static_pointer_cast<UnaryOp>(op);
            OpPtr operand = unary_op->get_operand();
            bw_toposort(operand);
            bw_order.push_back(op);
            break;
        }
        case Optype::BINARY: {
            std::shared_ptr<BinaryOp> binary_op = std::static_pointer_cast<BinaryOp>(op);
            OpPtr lhs = binary_op->get_lhs();
            OpPtr rhs = binary_op->get_rhs();
            bw_toposort(lhs);
            bw_toposort(rhs);
            bw_order.push_back(op);
            break;
        }
        case Optype::TRANSFORM: {
            std::shared_ptr<TransformOp> transform_op = std::static_pointer_cast<TransformOp>(op);
            OpPtr operand = transform_op->get_operand();
            bw_toposort(operand);
            bw_order.push_back(op);
            break;
        }
        default: {
            // Reduce operation
            std::shared_ptr<ReduceOp> reduce_op = std::static_pointer_cast<ReduceOp>(op);
            OpPtr operand = reduce_op->get_operand();
            bw_toposort(operand);
            bw_order.push_back(op);
            break;
        }
        }
    }

    void ComputeGraph::forward() {
        if (fw_order.empty()) {
            fw_toposort(output);
        }
    }

    void ComputeGraph::backward() {
        if (fw_order.empty()) {
            throw ComputeGraphNotForwardedException();
        }
        if (bw_order.empty()) {
            LazyPtr lazy = output->get_lazy();
            if (lazy->get_numel() > 1) {
                throw std::invalid_argument("Array " + lazy->get_id().str() + " must be a singleton to do gradient backpropation.");
            }
            // Initializes gradient with 1's
            output->init_grad(false);
            // Initializes the gradient array first without allocating buffers
            for (auto &op : std::views::reverse(fw_order)) {
                if (op->is_grad_enabled()) {
                    op->backward();
                }
            }
            // Order the gradient arrays
            for (auto &op : std::views::reverse(fw_order)) {
                // grad is null when backward is not implemented for op or that gradient is disabled
                if (op->grad_root != nullptr) {
                    bw_toposort(op->grad_root);
                }
            }
        }
    }

    const std::string ComputeGraph::str() const {
        if (fw_order.empty()) {
            throw ComputeGraphNotForwardedException();
        }
        LazyPtr lazy;
        std::string s = "Forward:\n";
        for (auto &op : fw_order) {
            lazy = op->get_lazy();
            s += op->str() + "\n";
        }
        s += "Backward:\n";
        for (auto &op : bw_order) {
            lazy = op->get_lazy();
            s += op->str() + "\n";
        }
        return s;
    }
} // namespace ax::graph