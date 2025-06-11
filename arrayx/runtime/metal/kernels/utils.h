#pragma once

#include <metal_stdlib>

typedef int64_t isize;

inline uint strided_idx(const uint id, const isize ndim, const constant isize *shape, const constant isize *stride) {
    isize carry = id;
    isize idx = 0;

    for (isize i = ndim - 1; i >= 0; i--) {
        idx += (carry % shape[i]) * stride[i];
        carry /= shape[i];
    }

    return idx;
}

template <class T>
struct Limits {
    static T finite_min() { return metal::numeric_limits<T>::min(); }
    static T finite_max() { return metal::numeric_limits<T>::max(); }
    static T min() { return metal::numeric_limits<T>::has_infinity ? -metal::numeric_limits<T>::infinity() : finite_min(); }
    static T max() { return metal::numeric_limits<T>::has_infinity ? metal::numeric_limits<T>::infinity() : finite_max(); }
};