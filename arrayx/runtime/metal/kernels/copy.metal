#include "utils.h"

template <class T, class R>
kernel void copy_vv(
    constant const uint *offset [[buffer(0)]],
    device T *input [[buffer(1)]],
    device R *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[1] + id] = input[offset[0] + id];
}

template <class T, class R>
kernel void copy_sv(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *output_stride [[buffer(3)]],
    device T *input [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[1] + output_idx] = input[offset[0] + id];
}

template <class T, class R>
kernel void copy_vs(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *input_stride [[buffer(3)]],
    device T *input [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint input_idx = strided_idx(id, ndim, shape, input_stride);
    output[offset[1] + id] = input[offset[0] + input_idx];
}

template <class T, class R>
kernel void copy_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *input_stride [[buffer(3)]],
    constant const int *output_stride [[buffer(4)]],
    device T *input [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint input_idx = strided_idx(id, ndim, shape, input_stride);
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[1] + output_idx] = input[offset[0] + input_idx];
}

#define make_copy(dtype, T, R) \
template [[host_name("copy_vv_" #dtype)]] [[kernel]] decltype(copy_vv<T, R>) copy_vv<T, R>; \
template [[host_name("copy_sv_" #dtype)]] [[kernel]] decltype(copy_sv<T, R>) copy_sv<T, R>; \
template [[host_name("copy_vs_" #dtype)]] [[kernel]] decltype(copy_vs<T, R>) copy_vs<T, R>; \
template [[host_name("copy_ss_" #dtype)]] [[kernel]] decltype(copy_ss<T, R>) copy_ss<T, R>;

make_copy(f32_f32, float, float);
make_copy(f32_i32, float, int);
make_copy(f32_b8, float, bool);
make_copy(i32_f32, int, float);
make_copy(i32_i32, int, int);
make_copy(i32_b8, int, bool);
make_copy(b8_f32, bool, float);
make_copy(b8_i32, bool, int);
make_copy(b8_b8, bool, bool);