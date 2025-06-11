#pragma once

#include "../../core/lazy.h"
#include "mtl_context.h"

namespace ax::runtime::metal {
    class CommandEncoder {
    private:
        std::shared_ptr<MTLContext> ctx;
        MTL::CommandBuffer *cmd_buff;
        MTL::ComputeCommandEncoder *encoder;
        isize buff_idx = 0;
        std::shared_ptr<MTLKernel> kernel;

    public:
        CommandEncoder(std::shared_ptr<MTLContext> ctx) : ctx(ctx) {
            cmd_buff = ctx->get_cmd_queue()->commandBuffer();
            encoder = cmd_buff->computeCommandEncoder();
        }

        CommandEncoder(const CommandEncoder &) = delete;
        ~CommandEncoder() = default;

        CommandEncoder &operator=(const CommandEncoder &) = delete;
        std::shared_ptr<MTLKernel> get_kernel() const { return kernel; }
        MTL::ComputeCommandEncoder *get_internal_encoder() const { return encoder; }

        void encode_buffer(const void *buff, isize size) {
            MTL::Buffer *mtl_buff = ctx->get_device()->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
            encoder->setBuffer(mtl_buff, 0, buff_idx++);
        }

        void encode_view(LazyPtr lazy) { encode_buffer(lazy->get_view().data(), sizeof(isize) * lazy->get_ndim()); }
        void encode_stride(LazyPtr lazy) { encode_buffer(lazy->get_stride().data(), sizeof(isize) * lazy->get_ndim()); }
        void encode_array(LazyPtr lazy) { encode_buffer(lazy->get_buff_ptr(), lazy->get_buff_nbytes()); }

        void set_pipeline_state(const std::string &kernel_name) {
            kernel = ctx->get_kernel(kernel_name);
            encoder->setComputePipelineState(kernel->get_state().get());
        }

        void dispatch_threads(isize nthreads) {
            MTL::Size grid_size = MTL::Size::Make(nthreads, 1, 1);
            isize max_threadgroup_size = kernel->get_state()->maxTotalThreadsPerThreadgroup();
            MTL::Size threadgroup_size = MTL::Size::Make(std::min(nthreads, max_threadgroup_size), 1, 1);
            dispatch_threads(grid_size, threadgroup_size);
        }

        void dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size) {
            encoder->dispatchThreads(grid_size, threadgroup_size);
            encoder->endEncoding();
            cmd_buff->commit();
        }

        void dispatch_threadgroups(MTL::Size threadgroup_count, MTL::Size threadgroup_size) {
            encoder->dispatchThreadgroups(threadgroup_count, threadgroup_size);
            encoder->endEncoding();
            cmd_buff->commit();
        }

        void wait_to_complete() { cmd_buff->waitUntilCompleted(); }

        double time_to_complete() {
            CFTimeInterval start = cmd_buff->GPUStartTime();
            cmd_buff->waitUntilCompleted();
            CFTimeInterval end = cmd_buff->GPUEndTime();
            return end - start;
        }
    };
} // namespace ax::runtime::metal