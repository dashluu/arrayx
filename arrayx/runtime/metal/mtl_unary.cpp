#include "mtl_runner.h"

namespace ax::runtime::metal
{
    void MTLRunner::run_unary_ss_kernel(const std::string &name, OpPtr in_op, OpPtr out_op)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyArrayPtr in_arr = in_op->get_lazy();
        LazyArrayPtr out_arr = out_op->get_lazy();
        bool strided_input = !in_arr->is_contiguous();
        bool strided_output = !out_arr->is_contiguous();

        if (strided_input || strided_output)
        {
            encoder.encode_ndim(in_arr);
        }
        encoder.encode_offset({in_arr, out_arr});
        if (strided_input || strided_output)
        {
            encoder.encode_view(in_arr);
        }
        if (strided_input)
        {
            encoder.encode_stride(in_arr);
        }
        if (strided_output)
        {
            encoder.encode_stride(out_arr);
        }

        encoder.encode_array(in_arr);
        encoder.encode_array(out_arr);
        std::string mode = std::string(strided_output ? "s" : "v") + std::string(strided_input ? "s" : "v");
        std::string kernel_name = name + "_" + mode + "_" + in_arr->get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(in_arr->get_numel());
        encoder.wait_to_complete();
        pool->release();
    }
}