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

struct Minimum
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs < rhs ? lhs : rhs; }
};

struct Maximum
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs > rhs ? lhs : rhs; }
};

template <class Op, class T, class R>
kernel void binary(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *shape [[buffer(2)]],
    const constant isize *lstride [[buffer(3)]],
    const constant isize *rstride [[buffer(4)]],
    const constant isize *outstride [[buffer(5)]],
    const constant bool *strided [[buffer(6)]],
    device T *lhs [[buffer(7)]],
    device T *rhs [[buffer(8)]],
    device R *output [[buffer(9)]],
    uint id [[thread_position_in_grid]])
{
    isize lidx = strided[0] ? strided_idx(id, ndim, shape, lstride) : id;
    isize ridx = strided[1] ? strided_idx(id, ndim, shape, rstride) : id;
    isize out_idx = strided[2] ? strided_idx(id, ndim, shape, outstride) : id;
    output[offset[2] + out_idx] = Op()(lhs[offset[0] + lidx], rhs[offset[1] + ridx]);
}

#define make_binary(opname, op, dtype, T, R) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary<op, T, R>) binary<op, T, R>;

#define make_cmp(opname, op, dtype, T) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(binary<op, T, bool>) binary<op, T, bool>;

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
binary(minimum, Minimum);
binary(maximum, Maximum);
cmp_all(eq, Eq);
cmp_all(neq, Neq);
numeric_cmp(lt, Lt);
numeric_cmp(gt, Gt);
numeric_cmp(leq, Leq);
numeric_cmp(geq, Geq);
