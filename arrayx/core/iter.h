#pragma once

#include "lazy_array.h"

namespace ax::core
{
    struct ArrayIter
    {
    private:
        std::shared_ptr<const LazyArray> arr;
        uint8_t *ptr;
        isize counter;

    public:
        ArrayIter(std::shared_ptr<const LazyArray> arr) : arr(arr)
        {
        }

        ArrayIter(const ArrayIter &) = delete;

        ArrayIter &operator=(const ArrayIter &) = delete;

        bool has_next() const { return counter < arr->get_shape().get_numel(); }

        isize count() const { return counter; }

        void start()
        {
            counter = 0;
        }

        uint8_t *next()
        {
            ptr = arr->strided_elm_ptr(counter);
            counter++;
            return ptr;
        }
    };
}