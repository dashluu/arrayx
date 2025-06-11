#include "mtl_runner.h"

namespace ax::runtime::metal {
    void MTLRunner::run_binary_kernel(const std::string &name, OpPtr lop, OpPtr rop, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr llazy = lop->get_lazy();
        LazyPtr rlazy = rop->get_lazy();
        LazyPtr out_lazy = out_op->get_lazy();
        isize ndim = llazy->get_ndim();
        isize offset[] = {llazy->get_offset(), rlazy->get_offset(), out_lazy->get_offset()};
        bool strided[] = {!llazy->is_contiguous(), !rlazy->is_contiguous(), !out_lazy->is_contiguous()};
        encoder.encode_buffer(&ndim, sizeof(isize));
        encoder.encode_buffer(offset, sizeof(isize) * 3);
        encoder.encode_view(llazy);
        encoder.encode_stride(llazy);
        encoder.encode_stride(rlazy);
        encoder.encode_stride(out_lazy);
        encoder.encode_buffer(strided, sizeof(bool) * 3);
        encoder.encode_array(llazy);
        encoder.encode_array(rlazy);
        encoder.encode_array(out_lazy);
        std::string kernel_name = name + "_" + llazy->get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(llazy->get_numel());
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace ax::runtime::metal