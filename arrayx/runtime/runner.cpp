#include "runner.h"

namespace ax::runtime {
    void Runner::run(OpPtr op) {
        switch (op->get_optype()) {
        case Optype::INITIALIZER: {
            run_initializer_op(op);
            break;
        }
        case Optype::UNARY: {
            run_unary_op(op);
            break;
        }
        case Optype::BINARY: {
            run_binary_op(op);
            break;
        }
        case Optype::MATMUL: {
            run_matmul_op(op);
            break;
        }
        case Optype::TRANSFORM: {
            run_transform_op(op);
            break;
        }
        default: {
            run_reduce_op(op);
            break;
        }
        }
    }

    void Runner::forward(std::shared_ptr<ComputePrimitive> primitive) {
        if (primitive->get_primitive_type() == ComputePrimitiveType::GRAPH) {
            auto graph = std::static_pointer_cast<ComputeGraph>(primitive);
            for (auto iter = graph->cbegin(); iter != graph->cend(); ++iter) {
                std::cout << (*iter)->str() << std::endl;
                run(*iter);
            }
        }
    }

    void Runner::backward(std::shared_ptr<ComputePrimitive> primitive) {
        if (primitive->get_primitive_type() == ComputePrimitiveType::GRAPH) {
            auto graph = std::static_pointer_cast<ComputeGraph>(primitive);
            for (auto iter = graph->crbegin(); iter != graph->crend(); ++iter) {
                run(*iter);
            }
        }
    }
} // namespace ax::runtime