#include <metal_stdlib>

template <class T>
kernel void full(
    device T *c [[buffer(0)]],
    device T *output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = *c;
}

template <class T>
kernel void arange(
    device int *start [[buffer(0)]],
    device int *step [[buffer(1)]],
    device T *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = *start + static_cast<int>(id) * *step;
}

#define make_initializer(opname, op, dtype, T) \
template [[host_name(#opname "_" #dtype)]] [[kernel]] decltype(op<T>) op<T>;

#define initializer_numeric(opname, op)     \
make_initializer(opname, op, f32, float);   \
make_initializer(opname, op, i32, int);     \

#define initializer_all(opname, op)     \
initializer_numeric(opname, op);        \
make_initializer(opname, op, b8, bool);

initializer_all(full, full);
initializer_numeric(arange, arange);