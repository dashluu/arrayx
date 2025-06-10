#pragma once

#include "allocator.h"

namespace ax::device {
    struct Buffer : public std::enable_shared_from_this<Buffer> {
    private:
        std::shared_ptr<Allocator> allocator = nullptr;
        uint8_t *ptr;
        isize nbytes;

        void free() {
            if (allocator != nullptr) {
                allocator->free(ptr, nbytes);
            }
        }

    public:
        Buffer(std::shared_ptr<Allocator> allocator, isize nbytes) : allocator(allocator), nbytes(nbytes) {
            ptr = allocator->alloc(nbytes);
        }

        Buffer(uint8_t *ptr, isize nbytes) : ptr(ptr), nbytes(nbytes) {}
        Buffer(const Buffer &buff) : ptr(buff.ptr), nbytes(buff.nbytes) {}
        ~Buffer() { free(); }

        Buffer &operator=(const Buffer &buff) {
            free();
            allocator = nullptr;
            ptr = buff.ptr;
            nbytes = buff.nbytes;
            return *this;
        }

        uint8_t *get_ptr() const { return ptr; }
        isize get_nbytes() const { return nbytes; }
    };
} // namespace ax::device