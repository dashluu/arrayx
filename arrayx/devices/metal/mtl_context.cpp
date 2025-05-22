#include "mtl_context.h"

namespace ax::devices::metal
{
    void MTLContext::init_kernel(const std::string &name)
    {
        auto kernel = std::make_shared<MTLKernel>(name);
        kernel->init(device, lib);
        kernel_by_name[name] = kernel;
    }

    void MTLContext::init_kernels(const std::vector<std::string> &opstrs, DtypePtrSet &dtypes, const std::vector<std::string> &modes)
    {
        for (auto &opstr : opstrs)
        {
            init_kernels(opstr, dtypes, modes);
        }
    }

    void MTLContext::init_kernels(const std::string &opstr, DtypePtrSet &dtypes, const std::vector<std::string> &modes)
    {
        for (auto &mode : modes)
        {
            for (auto &dtype : dtypes)
            {
                const std::string name = opstr + "_" + mode + "_" + dtype->get_name_str();
                init_kernel(name);
            }
        }
    }

    void MTLContext::init_kernels(const std::string &opstr, DtypePtrSet &dtypes)
    {
        for (auto &dtype : dtypes)
        {
            const std::string name = opstr + "_" + dtype->get_name_str();
            init_kernel(name);
        }
    }

    void MTLContext::init_initializer_kernels()
    {
        init_kernels("full", all_dtypes);
        init_kernels("arange", numeric_dtypes);
    }

    void MTLContext::init_unary_kernels()
    {
        std::vector<std::string> unary_opstrs = {"exp", "log", "neg", "recip", "sq", "sqrt"};
        std::vector<std::string> modes = {"vv", "sv", "vs", "ss"};
        init_kernels(unary_opstrs, numeric_dtypes, modes);
    }

    void MTLContext::init_binary_kernels()
    {
        std::vector<std::string> binary_opstrs = {"add", "sub", "mul", "div", "lt", "gt", "leq", "geq"};
        std::vector<std::string> eq_opstrs = {"eq", "neq"};
        std::vector<std::string> modes = {"vv", "sv", "vs", "ss"};
        init_kernels(binary_opstrs, numeric_dtypes, modes);
        init_kernels(eq_opstrs, all_dtypes, modes);
    }

    void MTLContext::init_reduce_kernels()
    {
        std::vector<std::string> reduce_opstrs = {"sum", "max", "min", "argmax", "argmin"};
        std::vector<std::string> modes = {"vv", "vs"};
        for (auto &opstr : reduce_opstrs)
        {
            init_kernels(opstr + "_all", numeric_dtypes, modes);
            init_kernels(opstr + "_col", numeric_dtypes, modes);
        }
    }

    void MTLContext::init_matmul_kernels()
    {
        init_kernels("matmul", numeric_dtypes, {"vv", "vs"});
    }

    void MTLContext::init_copy_kernels()
    {
        const std::vector<std::string> &modes = {"vv", "sv", "vs", "ss"};
        for (auto &dtype1 : all_dtypes)
        {
            for (auto &dtype2 : all_dtypes)
            {
                for (auto &mode : modes)
                {
                    const std::string name = "copy_" + mode + "_" + dtype1->get_name_str() + "_" + dtype2->get_name_str();
                    init_kernel(name);
                }
            }
        }
    }

    MTLContext::MTLContext(MTL::Device *mtl_device, const std::string &lib_path)
    {
        allocator = std::make_shared<MTLAllocator>();
        // device = NS::TransferPtr<MTL::Device>(MTL::CreateSystemDefaultDevice());
        device = NS::TransferPtr<MTL::Device>(mtl_device);
        NS::String *path = NS::String::string(lib_path.c_str(), NS::ASCIIStringEncoding);
        auto url = NS::URL::fileURLWithPath(path);
        // TODO: handle error
        NS::Error *error = nullptr;
        lib = NS::TransferPtr<MTL::Library>(device->newLibrary(url, &error));
        cmd_queue = NS::TransferPtr<MTL::CommandQueue>(device->newCommandQueue());
        // Initializes kernels here
        init_initializer_kernels();
        init_unary_kernels();
        init_binary_kernels();
        init_reduce_kernels();
        init_matmul_kernels();
        init_copy_kernels();
    }

    bool MTLContext::register_kernel(const std::string &name, std::shared_ptr<MTLKernel> kernel)
    {
        if (kernel_by_name.contains(name))
        {
            return false;
        }
        kernel_by_name.insert(std::make_pair(name, kernel));
        return true;
    }
}