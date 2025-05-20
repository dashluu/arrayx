#pragma once

#include "../utils.h"

namespace ax::core
{
    struct Buffer : public std::enable_shared_from_this<Buffer>
    {
    private:
        uint8_t *ptr;
        isize nbytes;
        bool root;

    public:
        Buffer(uint8_t *ptr, isize nbytes, bool root) : ptr(ptr), nbytes(nbytes), root(root)
        {
        }

        Buffer(const Buffer &buff) : ptr(buff.ptr), nbytes(buff.nbytes), root(false) {}

        Buffer &operator=(const Buffer &buff)
        {
            ptr = buff.ptr;
            nbytes = buff.nbytes;
            root = false;
            return *this;
        }

        uint8_t *get_ptr() const { return ptr; }

        isize get_nbytes() const { return nbytes; }

        bool is_root() { return root; }
    };
}