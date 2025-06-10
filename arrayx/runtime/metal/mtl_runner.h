#pragma once

#include "../runner.h"
#include "mtl_command_encoder.h"

namespace ax::runtime::metal {
    class MTLRunner : public Runner {
    protected:
        std::shared_ptr<MTLContext> ctx;

        void run_full_kernel(OpPtr op, isize c) override;
        void run_arange_kernel(OpPtr op, isize start, isize step) override;
        void run_binary_kernel(const std::string &name, OpPtr lop, OpPtr rop, OpPtr out_op) override;
        void run_matmul_kernel(OpPtr lop, OpPtr rop, OpPtr out_op) override;
        void run_unary_kernel(const std::string &name, OpPtr in_op, OpPtr out_op) override;
        void run_copy_kernel(OpPtr in_op, OpPtr out_op) override;
        void run_reduce_all_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, isize default_val) override;
        void run_reduce_col_kernel(const std::string &name, OpPtr in_op, OpPtr out_op, isize default_val) override;
        void run_initializer_op(OpPtr op) override;
        void run_unary_op(OpPtr op) override;
        void run_binary_op(OpPtr op) override;
        void run_transform_op(OpPtr op) override;

        template <class O>
        void run_simple_transform_op(OpPtr op) {
            auto transform_op = std::static_pointer_cast<O>(op);
            OpPtr operand = transform_op->get_operand();
            alloc(op->get_lazy(), operand->get_lazy());
        }

        void run_reduce_op(OpPtr op) override;
        void alloc(LazyPtr lazy) override { lazy->init_buff(std::make_shared<Buffer>(ctx->get_allocator(), lazy->get_nbytes())); }
        void alloc(LazyPtr out_lazy, LazyPtr in_lazy) override { out_lazy->init_buff(in_lazy->get_buff()); }

    public:
        MTLRunner(std::shared_ptr<MTLContext> ctx) : ctx(ctx) {}
    };
} // namespace ax::runtime::metal