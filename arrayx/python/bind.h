#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "../array/array.h"

namespace nb = nanobind;
namespace axc = ax::core;
namespace axd = ax::device;
namespace axr = ax::array;
using namespace nb::literals;

axc::DtypePtr dtype_from_nddtype(nb::dlpack::dtype nddtype)
{
	if (nddtype == nb::dtype<float>())
	{
		return &axc::f32;
	}
	else if (nddtype == nb::dtype<int>())
	{
		return &axc::i32;
	}
	return &axc::b8;
}

template <class T>
nb::ndarray<nb::numpy> array_to_numpy_impl(const axr::Array &arr)
{
	nb::object pyarr = nb::find(arr);
	std::vector<size_t> view(arr.get_shape().cbegin(), arr.get_shape().cend());

	return nb::ndarray<nb::numpy>(
		arr.get_ptr(),
		arr.get_ndim(),
		view.data(),
		pyarr.ptr(),
		arr.get_stride().data(),
		nb::dtype<T>(),
		// Numpy can only run on the cpu
		nb::device::cpu::value,
		'C');
}

nb::ndarray<nb::numpy> array_to_numpy(const axr::Array &arr);
axr::ArrayPtr array_from_numpy(nb::ndarray<nb::numpy> &ndarr);
