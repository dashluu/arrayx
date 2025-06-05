#pragma once

#include "../utils.h"

namespace ax::graph {
    enum class ComputePrimitiveType {
        GRAPH,
        KERNEL
    };

    class ComputePrimitive {
    protected:
        ComputePrimitiveType primitive_type;

    public:
        ComputePrimitive(ComputePrimitiveType primitive_type) : primitive_type(primitive_type) {}

        virtual ~ComputePrimitive() = default;

        ComputePrimitive(const ComputePrimitive &) = delete;

        ComputePrimitive &operator=(const ComputePrimitive &) = delete;

        ComputePrimitiveType get_primitive_type() const { return primitive_type; }
    };

    class ComputeKernel : public ComputePrimitive {
    protected:
        std::string name;

    public:
        ComputeKernel(const std::string &name) : ComputePrimitive(ComputePrimitiveType::KERNEL), name(name) {}

        const std::string &get_name() const { return name; }
    };
} // namespace ax::graph