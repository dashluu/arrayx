#pragma once

#include "metal.h"
#include "../../../utils.h"

namespace ax::runtime::metal
{
    struct MTLKernel : public std::enable_shared_from_this<MTLKernel>
    {
    private:
        std::string name;
        // shared ptr gets released once kernel is released
        NS::SharedPtr<MTL::Function> function;
        NS::SharedPtr<MTL::ComputePipelineState> state;

    public:
        MTLKernel(const std::string &name) : name(name) {}

        void init(NS::SharedPtr<MTL::Device> device, NS::SharedPtr<MTL::Library> lib)
        {
            NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
            auto ns_name = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
            function = NS::TransferPtr<MTL::Function>(lib->newFunction(ns_name));
            // TODO: handle error
            NS::Error *error = nullptr;
            state = NS::TransferPtr<MTL::ComputePipelineState>(device->newComputePipelineState(function.get(), &error));
            pool->release();
        }

        const std::string &get_name() { return name; }

        NS::SharedPtr<MTL::Function> get_function() { return function; }

        NS::SharedPtr<MTL::ComputePipelineState> get_state() { return state; }
    };
}