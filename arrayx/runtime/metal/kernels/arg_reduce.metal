#include "utils.h"

struct Argmax
{
    template <class T>
    void operator()(T lhs, T rhs, thread T *val, uint lidx, uint ridx, thread uint *out_idx) {
        if (lhs > rhs) {
            *val = lhs;
            *out_idx = lidx;
        } else {
            *val = rhs;
            *out_idx = ridx;
        }
    }

    template <class T>
    T get_default() { return Limits<T>::min(); }

    template <class T>
    static bool cmp(T old_val, T new_val) { return new_val > old_val; }
};

struct Argmin
{
    template <class T>
    void operator()(T lhs, T rhs, thread T *val, uint lidx, uint ridx, thread uint *out_idx) {
        if (lhs < rhs) {
            *val = lhs;
            *out_idx = lidx;
        } else {
            *val = rhs;
            *out_idx = ridx;
        }
    }

    template <class T>
    T get_default() { return Limits<T>::max(); }
    
    template <class T>
    static bool cmp(T old_val, T new_val) { return new_val < old_val; }
};

template <class Op, class T>
kernel void arg_reduce_all(
    const constant isize &numel [[buffer(0)]],
    const constant isize &ndim [[buffer(1)]],
    const constant isize *offset [[buffer(2)]],
    const constant isize *shape [[buffer(3)]],
    const constant isize *stride [[buffer(4)]],
    const constant bool *strided [[buffer(5)]],
    const device T *input [[buffer(6)]],
    device metal::_atomic<uint> *output [[buffer(7)]],
    threadgroup T *lvalue [[threadgroup(0)]],
    threadgroup uint *larg [[threadgroup(1)]],
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
    Op op;
    T default_val = op.template get_default<T>();
    isize in_idx = *strided ? strided_idx(gid, ndim, shape, stride) : gid;
    T val = input[offset[0] + in_idx];
    T shuffled_val;
    uint arg_idx = gid;
    uint shuffled_arg_idx;
    
    for (uint s = (lsize + simd_size - 1) / simd_size; s > 1; s /= simd_size)
    {
        // Perform per-SIMD partial reduction -> shuffling within SIMD group.
        // Each thread gets the value from another thread offset lanes above it.
        // Threads with index < offset lanes keep their original values.
        for (uint lanes = simd_size / 2; lanes > 0; lanes /= 2) {
            if (gid + lanes < numel) {
                shuffled_val = metal::simd_shuffle_down(val, lanes);
                shuffled_arg_idx = metal::simd_shuffle_down(arg_idx, lanes);
                op(val, shuffled_val, &val, arg_idx, shuffled_arg_idx, &arg_idx);
            }
        }
        
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            lvalue[simd_group_id] = val;
            larg[simd_group_id] = arg_idx;
        }
        
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lid < s) ? lvalue[lid] : default_val;
        arg_idx = (lid < s) ? larg[lid] : 0;
    }
    
    // Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
    for (uint lanes = simd_size / 2; lanes > 0; lanes /= 2) {
        if (gid + lanes < numel) {
            shuffled_val = metal::simd_shuffle_down(val, lanes);
            shuffled_arg_idx = metal::simd_shuffle_down(arg_idx, lanes);
            op(val, shuffled_val, &val, arg_idx, shuffled_arg_idx, &arg_idx);
        }
    }
    
    // Atomically update the reduction result.
    if (lid == 0) {
        T prev_val;
        uint prev_arg_idx = metal::atomic_load_explicit(output + offset[1], metal::memory_order_relaxed);
        
        do {
            in_idx = *strided ? strided_idx(prev_arg_idx, ndim, shape, stride) : prev_arg_idx;
            prev_val = input[offset[0] + in_idx];
            
            if (!Op::cmp(prev_val, val)) {
                break;
            }
        } while (!metal::atomic_compare_exchange_weak_explicit(output + offset[1], &prev_arg_idx, arg_idx, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
}

template <class Op, class T>
kernel void arg_reduce_col(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *shape [[buffer(2)]],
    const constant isize *stride [[buffer(3)]],
    const constant bool *strided [[buffer(4)]],
    const device T *input [[buffer(5)]],
    device metal::_atomic<uint> *output [[buffer(6)]],
    threadgroup T *lvalue [[threadgroup(0)]],
    threadgroup uint *larg [[threadgroup(1)]],
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
    Op op;
    T default_val = op.template get_default<T>();
    isize in_idx = *strided ? strided_idx(grow * N + gcol, ndim, shape, stride) : grow * N + gcol;
    T val = gcol < N ? input[offset[0] + in_idx] : default_val;
    T shuffled_val;
    uint arg_idx = gcol;
    uint shuffled_arg_idx;
    
    for (uint s = (lwidth + simd_size - 1) / simd_size; s > 1; s /= simd_size)
    {
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            if (gcol + lanes < N) {
                shuffled_val = metal::simd_shuffle_down(val, lanes);
                shuffled_arg_idx = metal::simd_shuffle_down(arg_idx, lanes);
                op(val, shuffled_val, &val, arg_idx, shuffled_arg_idx, &arg_idx);
            }
        }
        
        if (simd_lane_id == 0) {
            lvalue[lrow * lwidth + lcol / simd_size] = val;
            larg[lrow * lwidth + lcol / simd_size] = arg_idx;
        }
        
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lcol < s) ? lvalue[lrow * lwidth + lcol] : default_val;
        arg_idx = (lcol < s) ? larg[lrow * lwidth + lcol] : 0;
    }
    
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        if (gcol + lanes < N) {
            shuffled_val = metal::simd_shuffle_down(val, lanes);
            shuffled_arg_idx = metal::simd_shuffle_down(arg_idx, lanes);
            op(val, shuffled_val, &val, arg_idx, shuffled_arg_idx, &arg_idx);
        }
    }
    
    if (lcol == 0) {
        T prev_val;
        uint prev_arg_idx = metal::atomic_load_explicit(output + offset[1] + grow, metal::memory_order_relaxed);
        
        do {
            in_idx = *strided ? strided_idx(grow * N + prev_arg_idx, ndim, shape, stride) : grow * N + prev_arg_idx;
            prev_val = input[offset[0] + in_idx];
            
            if (!Op::cmp(prev_val, val)) {
                break;
            }
        } while (!metal::atomic_compare_exchange_weak_explicit(output + offset[1] + grow, &prev_arg_idx, arg_idx, metal::memory_order_relaxed, metal::memory_order_relaxed));
    }
}

#define make_arg_reduce(opname, op, dtype, T) \
template [[host_name(#opname "_all_" #dtype)]] [[kernel]] decltype(arg_reduce_all<op, T>) arg_reduce_all<op, T>;    \
template [[host_name(#opname "_col_" #dtype)]] [[kernel]] decltype(arg_reduce_col<op, T>) arg_reduce_col<op, T>;

#define arg_reduce(opname, op)              \
make_arg_reduce(opname, op, f32, float);    \
make_arg_reduce(opname, op, i32, int);

arg_reduce(argmax, Argmax);
arg_reduce(argmin, Argmin);