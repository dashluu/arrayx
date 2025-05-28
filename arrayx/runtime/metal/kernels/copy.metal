#include "utils.h"

template <class T, class R>
kernel void copy(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *instride [[buffer(3)]],
    constant const int *outstride [[buffer(4)]],
    constant const bool *strided [[buffer(5)]],
    device T *input [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint inidx = strided[0] ? strided_idx(id, ndim, shape, instride) : id;
    uint outidx = strided[1] ? strided_idx(id, ndim, shape, outstride) : id;
    output[offset[1] + outidx] = input[offset[0] + inidx];
}

#define make_copy(dtype, T, R) \
template [[host_name("copy_" #dtype)]] [[kernel]] decltype(copy<T, R>) copy<T, R>;

make_copy(f32_f32, float, float);
make_copy(f32_i32, float, int);
make_copy(f32_b8, float, bool);
make_copy(i32_f32, int, float);
make_copy(i32_i32, int, int);
make_copy(i32_b8, int, bool);
make_copy(b8_f32, bool, float);
make_copy(b8_i32, bool, int);
make_copy(b8_b8, bool, bool);