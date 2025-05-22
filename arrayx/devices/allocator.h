#pragma once

#include "../utils.h"

namespace ax::devices
{
    using ax::core::isize;

    struct Allocator : public std::enable_shared_from_this<Allocator>
    {
    protected:
        isize allocated = 0;

    public:
        Allocator() = default;
        Allocator(const Allocator &) = delete;
        virtual ~Allocator() = default;
        Allocator &operator=(const Allocator &) = delete;
        virtual uint8_t *alloc(isize nbytes) = 0;
        virtual void free(uint8_t *ptr, isize nbytes) = 0;
        isize get_allocated() const { return allocated; }
    };
}