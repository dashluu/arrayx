#include "mtl_runner.h"

namespace ax::runtime::metal {
    void MTLRunner::run_matmul_kernel(OpPtr lop, OpPtr rop, OpPtr out_op) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr llazy = lop->get_lazy();
        LazyPtr rlazy = rop->get_lazy();
        LazyPtr out_lazy = out_op->get_lazy();
        bool strided_input = !llazy->is_contiguous() || !rlazy->is_contiguous();

        // Encode buffers
        if (strided_input) {
            encoder.encode_ndim(llazy);
        }

        encoder.encode_offset({llazy, rlazy, out_lazy});
        encoder.encode_view(llazy);
        encoder.encode_view(rlazy);

        if (strided_input) {
            encoder.encode_stride(llazy);
            encoder.encode_stride(rlazy);
        }

        encoder.encode_array(llazy);
        encoder.encode_array(rlazy);
        encoder.encode_array(out_lazy);
        std::string mode = "v" + std::string(strided_input ? "s" : "v");
        std::string kernel_name = "matmul_" + mode + "_" + llazy->get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);

        const ShapeView &lhs_view = llazy->get_view();
        const ShapeView &rhs_view = rlazy->get_view();
        const isize batch_size = lhs_view[0];
        const isize nrow = lhs_view[1];
        const isize ncol = rhs_view[2];
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