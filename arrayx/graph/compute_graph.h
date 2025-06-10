#pragma once

#include "ops.h"

namespace ax::graph {
    class ComputeGraph : public std::enable_shared_from_this<ComputeGraph> {
    private:
        OpPtr output;
        std::unordered_set<Id> visited;
        std::vector<OpPtr> fw_order;
        std::vector<OpPtr> bw_order;

        void fw_toposort(OpPtr op);
        void bw_toposort(OpPtr op);

    public:
        ComputeGraph(OpPtr output) : output(output) {}
        OpPtr get_output() const { return output; }
        void forward();
        void backward();
        virtual void compile() = 0;
        const std::string str() const;
        std::vector<OpPtr>::const_iterator cbegin() const { return fw_order.cbegin(); }
        std::vector<OpPtr>::const_iterator cend() const { return fw_order.cend(); }
        std::vector<OpPtr>::const_iterator crbegin() const { return bw_order.cbegin(); }
        std::vector<OpPtr>::const_iterator crend() const { return bw_order.cend(); }
    };
} // namespace ax::graph