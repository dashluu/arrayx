#include "mtl_runner.h"

namespace ax::runtime::metal {
    void MTLRunner::run_unary_kernel(const std::string &name, OpPtr in_op, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = out_op->get_lazy();
        isize ndim = in_lazy->get_ndim();
        isize offset[] = {in_lazy->get_offset(), out_lazy->get_offset()};
        bool strided[] = {!in_lazy->is_contiguous(), !out_lazy->is_contiguous()};
        encoder.encode_buffer(&ndim, sizeof(isize));
        encoder.encode_buffer(offset, sizeof(isize) * 2);
        encoder.encode_view(in_lazy);
        encoder.encode_stride(in_lazy);
        encoder.encode_stride(out_lazy);
        encoder.encode_buffer(strided, sizeof(bool) * 2);
        encoder.encode_array(in_lazy);
        encoder.encode_array(out_lazy);
        std::string kernel_name = name + "_" + in_lazy->get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(in_lazy->get_numel());
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace ax::runtime::metal