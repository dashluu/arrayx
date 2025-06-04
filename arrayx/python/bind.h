#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "../nn/nn.h"

namespace ax::bind
{
}

namespace nb = nanobind;
namespace axc = ax::core;
namespace axd = ax::device;
namespace axr = ax::array;
namespace axnn = ax::nn;
namespace axb = ax::bind;
using namespace nb::literals;
