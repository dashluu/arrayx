#include "utils.h"

template <class T, class R>
kernel void matmul(
    const constant isize &ndim [[buffer(0)]],
    const constant isize *offset [[buffer(1)]],
    const constant isize *lshape [[buffer(2)]],
    const constant isize *rshape [[buffer(3)]],
    const constant isize *lstride [[buffer(4)]],
    const constant isize *rstride [[buffer(5)]],
    const constant bool *strided [[buffer(6)]],
    device T *lhs [[buffer(7)]],
    device T *rhs [[buffer(8)]],
    device R *output [[buffer(9)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z;
    const uint row = id.y;
    const uint col = id.x;
    // Get dimensions
    const isize B = lshape[0];  // Batch size
    const isize M = lshape[1];  // Rows in each matrix
    const isize N = rshape[2];  // Cols in each matrix
    const isize K = lshape[2];  // Inner dimension
    
    if (col < N && row < M && batch < B) {
        // Calculate output index
        // [batch, row, col] -> batch * (M * N) + row * N + col
        const isize out_idx = offset[2] + batch * M * N + row * N + col;
        R sum = 0;
        
        for (isize i = 0; i < K; i++) {
            // [batch, row, k] -> batch * (M * K) + row * K + k
            const isize lidx = offset[0] + (strided[0] ? strided_idx(batch * M * K + row * K + i, ndim, lshape, lstride) : batch * M * K + row * K + i);
            // [batch, k, col] -> batch * (K * N) + k * N + col
            const isize ridx = offset[1] + (strided[1] ? strided_idx(batch * K * N + N * i + col, ndim, rshape, rstride) : batch * K * N + N * i + col);
            sum += lhs[lidx] * rhs[ridx];
        }
        
        output[out_idx] = sum;
    }
}

#define make_matmul(dtype, T, R) \
template [[host_name("matmul_" #dtype)]] [[kernel]] decltype(matmul<T, R>) matmul<T, R>;

make_matmul(f32, float, float);
make_matmul(i32, int, int);