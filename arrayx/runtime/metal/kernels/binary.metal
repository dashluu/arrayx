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

template <class Op, class T, class R>
kernel void binary_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *lstride [[buffer(3)]],
    constant const int *rstride [[buffer(4)]],
    constant const int *outstride [[buffer(5)]],
    constant const bool *strided [[buffer(6)]],
    device T *lhs [[buffer(7)]],
    device T *rhs [[buffer(8)]],
    device R *output [[buffer(9)]],
    uint id [[thread_position_in_grid]])
{
    uint lidx = strided[0] ? strided_idx(id, ndim, shape, lstride) : id;
    uint ridx = strided[1] ? strided_idx(id, ndim, shape, rstride) : id;
    uint outidx = strided[2] ? strided_idx(id, ndim, shape, outstride) : id;
    output[offset[2] + outidx] = Op()(lhs[offset[0] + lidx], rhs[offset[1] + ridx]);
}

#define make_binary(opname, op, dtype, T, R) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary_ss<op, T, R>) binary_ss<op, T, R>;

#define make_cmp(opname, op, dtype, T) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary_ss<op, T, bool>) binary_ss<op, T, bool>;

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
