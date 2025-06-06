#include "iter.h"

namespace ax::core {
    uint8_t *LazyArray::strided_elm_ptr(isize k) const {
        if (is_contiguous()) {
            return get_ptr() + k * get_itemsize();
        }
        std::vector<isize> idx(get_ndim());
        isize carry = k;
        isize tmp;
        for (isize i = get_ndim() - 1; i >= 0; i--) {
            tmp = carry;
            idx[i] = tmp % shape[i];
            carry = tmp / shape[i];
        }
        const ShapeStride &stride = get_stride();
        uint8_t *ptr = get_ptr();
        for (size_t i = 0; i < idx.size(); i++) {
            ptr += idx[i] * stride[i] * get_itemsize();
        }
        return ptr;
    }

    const std::string LazyArray::str() const {
        auto iter = std::make_unique<ArrayIter>(shared_from_this());
        iter->start();
        bool next_elm_available = iter->has_next();
        if (!next_elm_available) {
            return "[]";
        }
        std::string s = "";
        for (isize i = 0; i < get_ndim(); i++) {
            s += "[";
        }
        ShapeView elms_per_dim = shape.get_elms_per_dim();
        size_t close = 0;
        while (next_elm_available) {
            close = 0;
            uint8_t *ptr = iter->next();
            // std::cout << std::hex << static_cast<void *>(ptr) << std::endl;
            s += dtype->get_value_as_str(ptr);
            for (ssize_t i = elms_per_dim.size() - 1; i >= 0; i--) {
                if (iter->count() % elms_per_dim[i] == 0) {
                    s += "]";
                    close += 1;
                }
            }
            next_elm_available = iter->has_next();
            if (next_elm_available) {
                if (close > 0) {
                    s += ", \n";
                } else {
                    s += ", ";
                }
                for (size_t i = 0; i < close; i++) {
                    s += "[";
                }
            }
        }
        return s;
    }
} // namespace ax::core