#include "utils.h"

struct Sum
{
    template <class T>
    void operator()(T lhs, T rhs, thread T *val) {
        *val =  lhs + rhs;
    }
};

struct Max
{
    template <class T>
    void operator()(T lhs, T rhs, thread T *val) {
        *val =  lhs > rhs ? lhs : rhs;
    }
};

struct Min
{
    template <class T>
    void operator()(T lhs, T rhs, thread T *val) {
        *val =  lhs < rhs ? lhs : rhs;
    }
};

struct AtomicSum
{
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T val)
    {
        // memory_order_relaxed guarantees atomicity without ordering or proper synchronization
        // since we're doing addition, this is somewhat similar to a counter
        // atomic_fetch_add_explicit runs output += val but atomically
        metal::atomic_fetch_add_explicit(output, val, metal::memory_order_relaxed);
    }
};

struct AtomicMaxInt {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val)
    {
        metal::atomic_fetch_max_explicit(output, new_val, metal::memory_order_relaxed);
    }
};

struct AtomicMaxFloat
{
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val)
    {
        // CAS algorithm
        // output = max(output, val)
        // Be cautious when T and R are not the same
        R old_val = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        do {
            if (old_val >= new_val) {
                break;
            }
        // old_val gets updated by metal::atomic_compare_exchange_weak_explicit if the operation fails
        // No need for old_val to be in the while loop
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
};

struct AtomicMinInt {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val)
    {
        metal::atomic_fetch_min_explicit(output, new_val, metal::memory_order_relaxed);
    }
};

struct AtomicMinFloat
{
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T new_val)
    {
        // CAS algorithm
        // output = min(output, val)
        // Be cautious when T and R are not the same
        R old_val = metal::atomic_load_explicit(output, metal::memory_order_relaxed);
        do {
            if (old_val <= new_val) {
                break;
            }
        // old_val gets updated by metal::atomic_compare_exchange_weak_explicit if the operation fails
        // No need for old_val to be in the while loop
        } while (!metal::atomic_compare_exchange_weak_explicit(output, &old_val, new_val, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
};

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_all_vv(
    const constant uint &numel [[buffer(0)]],
    const constant uint *offset [[buffer(1)]],
    const device T *input [[buffer(2)]],
    device metal::_atomic<R> *output [[buffer(3)]],
    const device R *default_val_ptr [[buffer(4)]],
    threadgroup R *ldata [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // Perform the first level of reduction.
    // Read from device memory, write to threadgroup memory.
    // val is stored in thread's register
    R default_val = *default_val_ptr;
    T val = input[offset[0] + gid];
    Op op;
    for (uint s = (lsize + simd_size - 1) / simd_size; s > 1; s /= simd_size)
    {
        // Perform per-SIMD partial reduction -> shuffling within SIMD group.
        // Each thread gets the value from another thread offset lanes above it.
        // Threads with index < offset lanes keep their original values.
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            if (gid + lanes < numel) {
                op(val, metal::simd_shuffle_down(val, lanes), &val);
            }
        }
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = val;
        }
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lid < s) ? ldata[lid] : default_val;
    }
    // Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        if (gid + lanes < numel) {
            op(val, metal::simd_shuffle_down(val, lanes), &val);
        }
    }
    // Atomically update the reduction result.
    if (lid == 0) {
        AtomicOp()(output + offset[1], val);
    }
}

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_all_vs(
    const constant uint &numel [[buffer(0)]],
    const constant uint &ndim [[buffer(1)]],
    const constant uint *offset [[buffer(2)]],
    const constant uint *shape [[buffer(3)]],
    const constant int *stride [[buffer(4)]],
    const device T *input [[buffer(5)]],
    device metal::_atomic<R> *output [[buffer(6)]],
    const device R *default_val_ptr [[buffer(7)]],
    threadgroup R *ldata [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // The algorithm is same as before with the exception that
    // elements are accessed non-contiguously
    uint idx = strided_idx(gid, ndim, shape, stride);
    R default_val = *default_val_ptr;
    T val = input[offset[0] + idx];
    Op op;
    for (uint s = (lsize + simd_size - 1) / simd_size; s > 1; s /= simd_size)
    {
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            if (gid + lanes < numel) {
                op(val, metal::simd_shuffle_down(val, lanes), &val);
            }
        }
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = val;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lid < s) ? ldata[lid] : default_val;
    }
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        if (gid + lanes < numel) {
            op(val, metal::simd_shuffle_down(val, lanes), &val);
        }
    }
    if (lid == 0) {
        AtomicOp()(output + offset[1], val);
    }
}

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_col_vv(
    const constant uint *offset [[buffer(0)]],
    const constant uint *shape [[buffer(1)]],
    const device T *input [[buffer(2)]],
    device metal::_atomic<R> *output [[buffer(3)]],
    device R *default_val_ptr [[buffer(4)]],
    threadgroup R *ldata [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint grow = gid.y;
    const uint gcol = gid.x;
    const uint lrow = lid.y;
    const uint lcol = lid.x;
    const uint lwidth = lsize.x;
    const uint N = shape[1];
    R default_val = *default_val_ptr;
    T val = gcol < N ? input[offset[0] + grow * N + gcol] : default_val;
    Op op;
    for (uint s = (lwidth + simd_size - 1) / simd_size; s > 1; s /= simd_size)
    {
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            if (gcol + lanes < N) {
                op(val, metal::simd_shuffle_down(val, lanes), &val);
            }
        }
        if (simd_lane_id == 0) {
            ldata[lrow * lwidth + lcol / simd_size] = val;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lcol < s) ? ldata[lrow * lwidth + lcol] : default_val;
    }
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        if (gcol + lanes < N) {
            op(val, metal::simd_shuffle_down(val, lanes), &val);
        }
    }
    if (lcol == 0) {
        AtomicOp()(output + offset[1] + grow, val);
    }
}

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_col_vs(
    const constant uint &ndim [[buffer(0)]],
    const constant uint *offset [[buffer(1)]],
    const constant uint *shape [[buffer(2)]],
    const constant int *stride [[buffer(3)]],
    const device T *input [[buffer(4)]],
    device metal::_atomic<R> *output [[buffer(5)]],
    const device R *default_val_ptr [[buffer(6)]],
    threadgroup R *ldata [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]])
{
    const uint grow = gid.y;
    const uint gcol = gid.x;
    const uint lrow = lid.y;
    const uint lcol = lid.x;
    const uint lwidth = lsize.x;
    const uint N = shape[1];
    // The algorithm is same as before with the exception that
    // elements are accessed non-contiguously
    uint idx = strided_idx(grow * N + gcol, ndim, shape, stride);
    R default_val = *default_val_ptr;
    T val = gcol < N ? input[offset[0] + idx] : default_val;
    Op op;
    for (uint s = (lwidth + simd_size - 1) / simd_size; s > 1; s /= simd_size)
    {
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            if (gcol + lanes < N) {
                op(val, metal::simd_shuffle_down(val, lanes), &val);
            }
        }
        if (simd_lane_id == 0) {
            ldata[lrow * lwidth + lcol / simd_size] = val;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lcol < s) ? ldata[lrow * lwidth + lcol] : default_val;
    }
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        if (gcol + lanes < N) {
            op(val, metal::simd_shuffle_down(val, lanes), &val);
        }
    }
    if (lcol == 0) {
        AtomicOp()(output + offset[1] + grow, val);
    }
}

#define make_reduce(opname, op, atomic_op, dtype, T, R) \
template [[host_name(#opname "_all_vv_" #dtype)]] [[kernel]] decltype(reduce_all_vv<op, atomic_op, T, R>) reduce_all_vv<op, atomic_op, T, R>;   \
template [[host_name(#opname "_all_vs_" #dtype)]] [[kernel]] decltype(reduce_all_vs<op, atomic_op, T, R>) reduce_all_vs<op, atomic_op, T, R>;   \
template [[host_name(#opname "_col_vv_" #dtype)]] [[kernel]] decltype(reduce_col_vv<op, atomic_op, T, R>) reduce_col_vv<op, atomic_op, T, R>;   \
template [[host_name(#opname "_col_vs_" #dtype)]] [[kernel]] decltype(reduce_col_vs<op, atomic_op, T, R>) reduce_col_vs<op, atomic_op, T, R>;


#define reduce(opname, op, atomic_op_float, atomic_op_int)      \
make_reduce(opname, op, atomic_op_float, f32, float, float);    \
make_reduce(opname, op, atomic_op_int, i32, int, int);

reduce(sum, Sum, AtomicSum, AtomicSum);
reduce(max, Max, AtomicMaxFloat, AtomicMaxInt);
reduce(min, Min, AtomicMinFloat, AtomicMinInt);