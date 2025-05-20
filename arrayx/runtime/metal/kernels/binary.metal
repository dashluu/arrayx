#include "utils.h"

struct Add
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

struct Sub
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs - rhs; }
};

struct Mul
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs * rhs; }
};

struct Div
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs / rhs; }
};

struct Eq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs == rhs; }
};

struct Neq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs != rhs; }
};

struct Lt
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs < rhs; }
};

struct Gt
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs > rhs; }
};

struct Leq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs <= rhs; }
};

struct Geq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs >= rhs; }
};

// Binary operations for scalar-scalar
template <class Op, class T, class R>
kernel void binary_ss_vv(
    constant const uint *offset [[buffer(0)]],
    device T *lhs [[buffer(1)]],
    device T *rhs [[buffer(2)]],
    device R *output [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[2] + id] = Op()(lhs[offset[0] + id], rhs[offset[1] + id]);
}

template <class Op, class T, class R>
kernel void binary_ss_sv(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *output_stride [[buffer(3)]],
    device T *lhs [[buffer(4)]],
    device T *rhs [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[2] + output_idx] = Op()(lhs[offset[0] + id], rhs[offset[1] + id]);
}

template <class Op, class T, class R>
kernel void binary_ss_vs(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *lhs_stride [[buffer(3)]],
    constant const int *rhs_stride [[buffer(4)]],
    device T *lhs [[buffer(5)]],
    device T *rhs [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint lhs_idx = strided_idx(id, ndim, shape, lhs_stride);
    uint rhs_idx = strided_idx(id, ndim, shape, rhs_stride);
    output[offset[2] + id] = Op()(lhs[offset[0] + lhs_idx], rhs[offset[1] + rhs_idx]);
}

template <class Op, class T, class R>
kernel void binary_ss_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *lhs_stride [[buffer(3)]],
    constant const int *rhs_stride [[buffer(4)]],
    constant const int *output_stride [[buffer(5)]],
    device T *lhs [[buffer(6)]],
    device T *rhs [[buffer(7)]],
    device R *output [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    uint lhs_idx = strided_idx(id, ndim, shape, lhs_stride);
    uint rhs_idx = strided_idx(id, ndim, shape, rhs_stride);
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[2] + output_idx] = Op()(lhs[offset[0] + lhs_idx], rhs[offset[1] + rhs_idx]);
}

#define make_binary(opname, op, dtype, T, R) \
template [[host_name(#opname "_vv_" #dtype)]] [[kernel]] decltype(binary_ss_vv<op, T, R>) binary_ss_vv<op, T, R>;   \
template [[host_name(#opname "_sv_" #dtype)]] [[kernel]] decltype(binary_ss_sv<op, T, R>) binary_ss_sv<op, T, R>;   \
template [[host_name(#opname "_vs_" #dtype)]] [[kernel]] decltype(binary_ss_vs<op, T, R>) binary_ss_vs<op, T, R>;   \
template [[host_name(#opname "_ss_" #dtype)]] [[kernel]] decltype(binary_ss_ss<op, T, R>) binary_ss_ss<op, T, R>;

#define make_cmp(opname, op, dtype, T) \
template [[host_name(#opname "_vv_" #dtype)]] [[kernel]] decltype(binary_ss_vv<op, T, bool>) binary_ss_vv<op, T, bool>; \
template [[host_name(#opname "_sv_" #dtype)]] [[kernel]] decltype(binary_ss_sv<op, T, bool>) binary_ss_sv<op, T, bool>; \
template [[host_name(#opname "_vs_" #dtype)]] [[kernel]] decltype(binary_ss_vs<op, T, bool>) binary_ss_vs<op, T, bool>; \
template [[host_name(#opname "_ss_" #dtype)]] [[kernel]] decltype(binary_ss_ss<op, T, bool>) binary_ss_ss<op, T, bool>;

#define binary(opname, op)                  \
make_binary(opname, op, f32, float, float); \
make_binary(opname, op, i32, int, int);

#define numeric_cmp(opname, op)     \
make_cmp(opname, op, f32, float);   \
make_cmp(opname, op, i32, int);

#define cmp_all(opname, op)     \
numeric_cmp(opname, op);        \
make_cmp(opname, op, b8, bool);

binary(add, Add);
binary(sub, Sub);
binary(mul, Mul);
binary(div, Div);
cmp_all(eq, Eq);
cmp_all(neq, Neq);
numeric_cmp(lt, Lt);
numeric_cmp(gt, Gt);
numeric_cmp(leq, Leq);
numeric_cmp(geq, Geq);
