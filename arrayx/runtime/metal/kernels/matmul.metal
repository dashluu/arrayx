#include "utils.h"

template <class T, class R>
kernel void matmul_vv(
    const constant uint *offset [[buffer(0)]],
    const constant uint *lhs_shape [[buffer(1)]],
    const constant uint *rhs_shape [[buffer(2)]],
    device T *lhs [[buffer(3)]],
    device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z;
    const uint row = id.y;
    const uint col = id.x;
    // Get dimensions
    const uint B = lhs_shape[0];  // Batch size
    const uint M = lhs_shape[1];  // Rows in each matrix
    const uint N = rhs_shape[2];  // Cols in each matrix
    const uint K = lhs_shape[2];  // Inner dimension
    if (col < N && row < M && batch < B) {
        // Calculate output index
        // [batch, row, col] -> batch * (M * N) + row * N + col
        const uint out_idx = offset[2] + batch * M * N + row * N + col;
        R sum = 0;
        for (uint i = 0; i < K; i++) {
            // [batch, row, k] -> batch * (M * K) + row * K + k
            const uint lhs_idx = offset[0] + batch * M * K + row * K + i;
            // [batch, k, col] -> batch * (K * N) + k * N + col
            const uint rhs_idx = offset[1] + batch * K * N + N * i + col;
            sum += lhs[lhs_idx] * rhs[rhs_idx];
        }
        output[out_idx] = sum;
    }
}

template <class T, class R>
kernel void matmul_vs(
    const constant uint &ndim [[buffer(0)]],
    const constant uint *offset [[buffer(1)]],
    const constant uint *lhs_shape [[buffer(2)]],
    const constant uint *rhs_shape [[buffer(3)]],
    const constant int *lhs_stride [[buffer(4)]],
    const constant int *rhs_stride [[buffer(5)]],
    device T *lhs [[buffer(6)]],
    device T *rhs [[buffer(7)]],
    device R *output [[buffer(8)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z;
    const uint row = id.y;
    const uint col = id.x;
    // Get dimensions
    const uint B = lhs_shape[0];  // Batch size
    const uint M = lhs_shape[1];  // Rows in each matrix
    const uint N = rhs_shape[2];  // Cols in each matrix
    const uint K = lhs_shape[2];  // Inner dimension
    if (col < N && row < M && batch < B) {
        // Calculate output index
        // [batch, row, col] -> batch * (M * N) + row * N + col
        const uint out_idx = offset[2] + batch * M * N + row * N + col;
        R sum = 0;
        for (uint i = 0; i < K; i++) {
            // [batch, row, k] -> batch * (M * K) + row * K + k
            const uint lhs_idx = offset[0] + strided_idx(batch * M * K + row * K + i, ndim, lhs_shape, lhs_stride);
            // [batch, k, col] -> batch * (K * N) + k * N + col
            const uint rhs_idx = offset[1] + strided_idx(batch * K * N + N * i + col, ndim, rhs_shape, rhs_stride);
            sum += lhs[lhs_idx] * rhs[rhs_idx];
        }
        output[out_idx] = sum;
    }
}

#define make_matmul(dtype, T, R) \
template [[host_name("matmul_vv_" #dtype)]] [[kernel]] decltype(matmul_vv<T, R>) matmul_vv<T, R>;   \
template [[host_name("matmul_vs_" #dtype)]] [[kernel]] decltype(matmul_vs<T, R>) matmul_vs<T, R>;

make_matmul(f32, float, float);
make_matmul(i32, int, int);