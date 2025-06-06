#include "array.h"

namespace ax::array {
    void Array::eval() {
        if (compute_graph == nullptr) {
            compute_graph = get_backend_graph_builder()(op);
            compute_graph->forward();
            std::cout << compute_graph->str() << std::endl;
            get_backend_runner()->forward(compute_graph);
        } else if (!op->is_idempotent()) {
            get_backend_runner()->forward(compute_graph);
        }
    }

    void Array::backward() {
        eval();
        compute_graph->backward();
        get_backend_runner()->backward(compute_graph);
    }
} // namespace ax::array
