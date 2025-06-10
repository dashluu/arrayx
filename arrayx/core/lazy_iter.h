#pragma once

#include "lazy.h"

namespace ax::core {
    struct LazyIter {
    private:
        std::shared_ptr<const Lazy> lazy;
        uint8_t *ptr;
        isize counter;

    public:
        LazyIter(std::shared_ptr<const Lazy> lazy) : lazy(lazy) {}

        LazyIter(const LazyIter &) = delete;

        LazyIter &operator=(const LazyIter &) = delete;

        bool has_next() const { return counter < lazy->get_shape().get_numel(); }

        isize count() const { return counter; }

        void start() {
            counter = 0;
        }

        uint8_t *next() {
            ptr = lazy->strided_elm_ptr(counter);
            counter++;
            return ptr;
        }
    };
} // namespace ax::core