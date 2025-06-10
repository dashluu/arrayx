#pragma once

#include "../nn/jit.h"
#include "../nn/nn.h"
#include "../nn/optim.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

namespace ax::bind {
}

namespace nb = nanobind;
namespace axc = ax::core;
namespace axd = ax::device;
namespace axr = ax::array;
namespace axnn = ax::nn;
namespace axo = ax::optim;
namespace axb = ax::bind;
using namespace nb::literals;
