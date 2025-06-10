#include "array.h"

namespace ax::array {
    void Array::eval() {
        if (compute_graph == nullptr) {
            compute_graph = get_backend_graph_builder()(op);
            // TODO: use compute_graph->compile()
            compute_graph->forward();
            get_backend_runner()->forward(compute_graph);
            initial_run = true;
        } else if (!initial_run || !op->is_idempotent()) {
            get_backend_runner()->forward(compute_graph);
            initial_run = true;
        }
    }

    void Array::backward() {
        eval();
        compute_graph->backward();
        get_backend_runner()->backward(compute_graph);
    }

    void Array::compile() {
        if (compute_graph == nullptr) {
            compute_graph = get_backend_graph_builder()(op);
            // TODO: use compute_graph->compile()
            compute_graph->forward();
        }
    }
} // namespace ax::array
