#pragma once

#include "../compute_graph.h"

namespace ax::graph::metal {
    class MTLGraph : public ComputeGraph {

    public:
        MTLGraph(OpPtr output) : ComputeGraph(output) {}

        void compile() override {
            // TODO: implement this later
        }
    };
} // namespace ax::graph::metal