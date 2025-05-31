#pragma once

#include <metal_stdlib>

#define MAX_NDIM 8

uint strided_idx(const uint id, const uint ndim, const constant uint *shape, const constant int *stride);