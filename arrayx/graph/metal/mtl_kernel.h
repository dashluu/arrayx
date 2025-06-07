#pragma once

#include "../../utils.h"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace ax::graph::metal {
    class MTLKernel : public std::enable_shared_from_this<MTLKernel> {
    private:
        // shared ptr gets released once kernel is released
        NS::SharedPtr<MTL::Function> function;
        NS::SharedPtr<MTL::ComputePipelineState> state;
        std::string name;

    public:
        MTLKernel(const std::string &name) : name(name) {}

        void init(NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::Library> lib) {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            auto ns_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
            function = NS::TransferPtr<MTL::Function>(lib->newFunction(ns_name));
            // TODO: handle error
            NS::Error *error = nullptr;
            state = NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(function.get(), &error));
            pool->release();
        }

        NS::SharedPtr<MTL::Function> get_function() const { return function; }

        NS::SharedPtr<MTL::ComputePipelineState> get_state() const { return state; }
    };
} // namespace ax::graph::metal