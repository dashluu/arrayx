#include "utils.h"

template <class T, class R>
kernel void copy(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *shape [[buffer(2)]],
    const constant isize *instride [[buffer(3)]],
    const constant isize *outstride [[buffer(4)]],
    const constant bool *strided [[buffer(5)]],
    device T *input [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    isize in_idx = strided[0] ? strided_idx(id, ndim, shape, instride) : id;
    isize out_idx = strided[1] ? strided_idx(id, ndim, shape, outstride) : id;
    output[offset[1] + out_idx] = input[offset[0] + in_idx];
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