#include "mtl_runner.h"

namespace ax::runtime::metal {
    void MTLRunner::run_matmul_kernel(OpPtr lop, OpPtr rop, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr llazy = lop->get_lazy();
        LazyPtr rlazy = rop->get_lazy();
        LazyPtr out_lazy = out_op->get_lazy();
        isize ndim = llazy->get_ndim();
        isize offset[] = {llazy->get_offset(), rlazy->get_offset(), out_lazy->get_offset()};
        bool strided[] = {!llazy->is_contiguous(), !rlazy->is_contiguous()};
        encoder.encode_buffer(&ndim, sizeof(isize));
        encoder.encode_buffer(offset, sizeof(isize) * 3);
        encoder.encode_view(llazy);
        encoder.encode_view(rlazy);
        encoder.encode_stride(llazy);
        encoder.encode_stride(rlazy);
        encoder.encode_buffer(strided, sizeof(bool) * 2);
        encoder.encode_array(llazy);
        encoder.encode_array(rlazy);
        encoder.encode_array(out_lazy);
        std::string kernel_name = "matmul_" + llazy->get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);

        const ShapeView &lview = llazy->get_view();
        const ShapeView &rview = rlazy->get_view();
        const isize batch_size = lview[0];
        const isize nrow = lview[1];
        const isize ncol = rview[2];
        const isize x_threads_per_group = 8;
        const isize y_threads_per_group = 8;
        const isize z_threads_per_group = 4;
        // Even if matrix is smaller than one threadgroup, we still need at least 1 group
        const isize x_group_count = std::max(1ll, (ncol + x_threads_per_group - 1) / x_threads_per_group);
        const isize y_group_count = std::max(1ll, (nrow + y_threads_per_group - 1) / y_threads_per_group);
        const isize z_group_count = std::max(1ll, (batch_size + z_threads_per_group - 1) / z_threads_per_group);
        // Compute # threadgroups and threadgroup size
        auto threadgroup_count = MTL::Size::Make(x_group_count, y_group_count, z_group_count);
        auto threadgroup_size = MTL::Size::Make(x_threads_per_group, y_threads_per_group, z_threads_per_group);

        // Dispatch kernel
        encoder.dispatch_threadgroups(threadgroup_count, threadgroup_size);
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace ax::runtime::metal