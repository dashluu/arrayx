#include "mtl_runner.h"

namespace ax::runtime::metal {
    void MTLRunner::run_full_kernel(OpPtr op, isize c) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr lazy = op->get_lazy();
        DtypePtr dtype = lazy->get_dtype();
        encoder.encode_buffer(&c, dtype->get_size());
        encoder.encode_array(lazy);
        std::string kernel_name = "full_" + dtype->str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(lazy->get_numel());
        encoder.wait_to_complete();
        pool->release();
    }

    void MTLRunner::run_arange_kernel(OpPtr op, isize start, isize step) {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        LazyPtr lazy = op->get_lazy();
        encoder.encode_buffer(&start, sizeof(isize));
        encoder.encode_buffer(&step, sizeof(isize));
        encoder.encode_array(lazy);
        std::string kernel_name = "arange_" + lazy->get_dtype()->str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(lazy->get_numel());
        encoder.wait_to_complete();
        pool->release();
    }
} // namespace ax::runtime::metal