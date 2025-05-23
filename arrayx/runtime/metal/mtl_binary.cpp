#include "mtl_runner.h"

namespace ax::runtime::metal
{
    void MTLRunner::run_binary_ss_kernel(const std::string &name, OpPtr lop, OpPtr rop, OpPtr out_op)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyArrayPtr larr = lop->get_lazy();
        LazyArrayPtr rarr = rop->get_lazy();
        LazyArrayPtr out_arr = out_op->get_lazy();
        bool strided_input = !larr->is_contiguous() || !rarr->is_contiguous();
        bool strided_output = !out_arr->is_contiguous();

        if (strided_input || strided_output)
        {
            encoder.encode_ndim(larr);
        }
        encoder.encode_offset({larr, rarr, out_arr});
        if (strided_input || strided_output)
        {
            encoder.encode_view(larr);
        }
        if (strided_input)
        {
            encoder.encode_stride(larr);
            encoder.encode_stride(rarr);
        }
        if (strided_output)
        {
            encoder.encode_stride(out_arr);
        }

        encoder.encode_array(larr);
        encoder.encode_array(rarr);
        encoder.encode_array(out_arr);
        std::string mode = std::string(strided_output ? "s" : "v") + std::string(strided_input ? "s" : "v");
        std::string kernel_name = name + "_" + mode + "_" + larr->get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(larr->get_numel());
        encoder.wait_to_complete();
        pool->release();
    }
}