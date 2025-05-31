#include "utils.h"

uint strided_idx(const uint id, const uint ndim, const constant uint *shape, const constant int *stride)
{
    uint dim[MAX_NDIM] = {0};
    uint carry = id;
    for (int i = ndim - 1; i >= 0; i--)
    {
        dim[i] = carry % shape[i];
        carry /= shape[i];
    }
    uint idx = 0;
    for (uint i = 0; i < ndim; i++)
    {
        idx += dim[i] * stride[i];
    }
    return idx;
}