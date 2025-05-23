#pragma once

#include "../../core/dtype.h"
#include "../../device/metal/mtl_allocator.h"
#include "../runner_context.h"
#include "mtl_kernel.h"

namespace ax::runtime::metal
{
    using namespace ax::core;
    using namespace ax::device::metal;

    class MTLContext : public RunnerContext
    {
    private:
        std::shared_ptr<MTLAllocator> allocator;
        NS::SharedPtr<MTL::Device> device;
        NS::SharedPtr<MTL::Library> lib;
        NS::SharedPtr<MTL::CommandQueue> cmd_queue;
        std::unordered_map<std::string, std::shared_ptr<MTLKernel>> kernel_by_name;

        void init_kernel(const std::string &name);
        void init_kernels(const std::vector<std::string> &opstrs, DtypePtrSet &dtypes, const std::vector<std::string> &modes);
        void init_kernels(const std::string &opstr, DtypePtrSet &dtypes, const std::vector<std::string> &modes);
        void init_kernels(const std::string &opstr, DtypePtrSet &dtypes);
        void init_initializer_kernels();
        void init_unary_kernels();
        void init_binary_kernels();
        void init_reduce_kernels();
        void init_matmul_kernels();
        void init_copy_kernels();

    public:
        MTLContext(MTL::Device *mtl_device, const std::string &lib_path);

        bool register_kernel(const std::string &name, std::shared_ptr<MTLKernel> kernel);

        std::shared_ptr<MTLAllocator> get_allocator() const
        {
            return allocator;
        }

        std::shared_ptr<MTLKernel> get_kernel(const std::string &name) const
        {
            return kernel_by_name.at(name);
        }

        NS::SharedPtr<MTL::Device> get_device() const
        {
            return device;
        }

        NS::SharedPtr<MTL::CommandQueue> get_cmd_queue() const
        {
            return cmd_queue;
        }
    };
}