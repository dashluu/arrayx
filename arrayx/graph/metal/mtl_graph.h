#pragma once

#include "../compute_graph.h"

namespace ax::graph::metal {
    class MTLGraph : public ComputeGraph {

    public:
        MTLGraph(OpPtr output) : ComputeGraph(output) {}

        virtual std::shared_ptr<ComputeKernel> compile() override {
            // TODO: implement this later
            return nullptr;
        }
    };
} // namespace ax::graph::metal