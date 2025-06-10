#include "mtl_runner.h"

namespace ax::runtime::metal {
    void MTLRunner::run_reduce_all_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, isize default_val) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = out_op->get_lazy();
        bool strided_input = !in_lazy->is_contiguous();

        // Encode buffers
        isize numel = in_lazy->get_numel();
        encoder.encode_buffer(&numel, sizeof(mtl_usize), false);
        if (strided_input) {
            encoder.encode_ndim(in_lazy);
        }
        encoder.encode_offset({in_lazy, out_lazy});
        if (strided_input) {
            encoder.encode_view(in_lazy);
            encoder.encode_stride(in_lazy);
        }
        encoder.encode_array(in_lazy);
        encoder.encode_array(out_lazy);
        DtypePtr dtype = in_lazy->get_dtype();
        encoder.encode_buffer(&default_val, dtype->get_size(), false);

        // Configure kernel
        std::string mode = "v" + std::string(strided_input ? "s" : "v");
        std::string kernel_name = name + "_all_" + mode + "_" + dtype->str();
        encoder.set_pipeline_state(kernel_name);

        // Calculate optimal thread configuration
        const isize max_threadgroup_size = encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        const isize simd_size = encoder.get_kernel()->get_state()->threadExecutionWidth();
        const isize threadgroup_size = std::min(numel, max_threadgroup_size);
        // Set threadgroup memory size
        const isize val_threadgroup_nbytes = threadgroup_size * dtype->get_size();
        const isize arg_threadgroup_nbytes = threadgroup_size * sizeof(mtl_usize);
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(val_threadgroup_nbytes, 0);
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(arg_threadgroup_nbytes, 1);

        // Dispatch kernel
        encoder.dispatch_threads(numel);
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRunner::run_reduce_col_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, isize default_val) {
        // Initialize Metal autorelease pool and encoder
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = out_op->get_lazy();
        bool strided_input = !in_lazy->is_contiguous();

        // Encode buffers
        if (strided_input) {
            encoder.encode_ndim(in_lazy);
        }
        encoder.encode_offset({in_lazy, out_lazy});
        encoder.encode_view(in_lazy);
        if (strided_input) {
            encoder.encode_stride(in_lazy);
        }
        encoder.encode_array(in_lazy);
        encoder.encode_array(out_lazy);
        DtypePtr dtype = in_lazy->get_dtype();
        encoder.encode_buffer(&default_val, dtype->get_size(), false);

        // Configure kernel
        std::string mode = "v" + std::string(strided_input ? "s" : "v");
        std::string kernel_name = name + "_col_" + mode + "_" + dtype->str();
        encoder.set_pipeline_state(kernel_name);

        // Calculate optimal thread configuration
        const isize max_threadgroup_size = encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        const isize simd_size = encoder.get_kernel()->get_state()->threadExecutionWidth();
        const ShapeView &view = in_lazy->get_view();
        const isize nrow = view[0];
        const isize ncol = (view[1] + simd_size - 1) / simd_size * simd_size;
        const isize col_threadgroup_size = std::min(ncol, max_threadgroup_size);
        const isize row_threadgroup_size = std::min(nrow, max_threadgroup_size / col_threadgroup_size);
        const isize val_threadgroup_nbytes = col_threadgroup_size * row_threadgroup_size * dtype->get_size();
        const isize arg_threadgroup_nbytes = col_threadgroup_size * row_threadgroup_size * sizeof(mtl_usize);
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(val_threadgroup_nbytes, 0);
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(arg_threadgroup_nbytes, 1);
        MTL::Size grid_size = MTL::Size::Make(ncol, nrow, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(col_threadgroup_size, row_threadgroup_size, 1);

        // Dispatch kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace ax::runtime::metal