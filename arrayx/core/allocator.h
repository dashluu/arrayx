#pragma once

#include "../utils.h"
#include "buffer.h"

namespace ax::core
{
    struct Allocator
    {
    protected:
        isize allocated = 0;

    public:
        Allocator() = default;
        Allocator(const Allocator &) = delete;
        virtual ~Allocator() = default;
        Allocator &operator=(const Allocator &) = delete;
        virtual std::shared_ptr<Buffer> alloc(isize nbytes) = 0;
        virtual void free(std::shared_ptr<Buffer> buff) = 0;
        isize get_allocated() const { return allocated; }
    };

    struct CommonAllocator : public Allocator
    {
        std::shared_ptr<Buffer> alloc(isize nbytes) override
        {
            allocated += nbytes;
            auto ptr = new uint8_t[nbytes];
            std::memset(ptr, 0, nbytes);
            return std::make_shared<Buffer>(ptr, nbytes, true);
        }

        void free(std::shared_ptr<Buffer> buff) override
        {
            if (buff->is_root())
            {
                allocated -= buff->get_nbytes();
                delete[] buff->get_ptr();
            }
        }
    };

#if __APPLE__
    // Allocator for both CPU and MPS
    inline CommonAllocator allocator0;
#else
    // TODO: implement for other platforms such as CUDA
#endif
}