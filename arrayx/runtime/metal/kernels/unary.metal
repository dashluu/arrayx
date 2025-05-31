#include "utils.h"

struct Exp
{
    template <typename T>
    float operator()(T x) const
    {
        return metal::exp(static_cast<float>(x));
    }
};

struct Log
{
    template <typename T>
    float operator()(T x) const
    {
        return metal::log(static_cast<float>(x));
    }
};

struct Neg
{
    template <typename T>
    T operator()(T x) const
    {
        return -x;
    }
};

struct Recip
{
    template <typename T>
    float operator()(T x) const
    {
        return 1.0f / x;
    }
};

struct Sqrt
{
    template <typename T>
    float operator()(T x) const
    {
        return metal::sqrt(static_cast<float>(x));
    }
};

struct Sq
{
    template <typename T>
    float operator()(T x) const
    {
        return x * x;
    }
};

template <class Op, class T, class R>
kernel void unary(
    const constant uint &ndim [[buffer(0)]],
    const constant uint *offset [[buffer(1)]],
    const constant uint *shape [[buffer(2)]],
    const constant int *instride [[buffer(3)]],
    const constant int *outstride [[buffer(4)]],
    const constant bool *strided [[buffer(5)]],
    device T *input [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint inidx = strided[0] ? strided_idx(id, ndim, shape, instride) : id;
    uint outidx = strided[1] ? strided_idx(id, ndim, shape, outstride) : id;
    output[offset[1] + outidx] = Op()(input[offset[0] + inidx]);
}

#define make_unary_all(opname, op, dtype, T, R) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(unary<op, T, R>) unary<op, T, R>;

#define make_unary_float(opname, op, dtype, T) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(unary<op, T, float>) unary<op, T, float>;

#define unary_float(opname, op)             \
make_unary_float(opname, op, f32, float);   \
make_unary_float(opname, op, i32, int);

#define unary_all(opname, op)                   \
make_unary_all(opname, op, f32, float, float);  \
make_unary_all(opname, op, i32, int, int);

unary_all(exp, Exp);
unary_float(log, Log);
unary_all(neg, Neg);
unary_float(recip, Recip);
unary_all(sq, Sq);
unary_float(sqrt, Sqrt);
