#pragma once

#include "bind.h"

namespace ax::bind
{
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

	template <class T>
	nb::ndarray<nb::pytorch> array_to_torch_impl(const axr::Array &arr)
	{
		nb::object pyarr = nb::find(arr);
		std::vector<size_t> view(arr.get_shape().cbegin(), arr.get_shape().cend());
		int device;

		switch (arr.get_device()->get_type())
		{
		case axd::DeviceType::CPU:
			device = nb::device::cpu::value;
			break;
		default:
			// Try CPU for now
			// TODO: change to metal later
			device = nb::device::metal::value;
			break;
		}

		return nb::ndarray<nb::pytorch>(
			arr.get_ptr(),
			arr.get_ndim(),
			view.data(),
			pyarr.ptr(),
			arr.get_stride().data(),
			nb::dtype<T>(),
			device,
			'C');
	}

	inline std::string get_pyclass(const nb::object &obj)
	{
		auto cls = obj.attr("__class__");
		auto name = cls.attr("__name__");
		return nb::cast<std::string>(name);
	}

	axc::isize get_pyidx(axc::isize len, axc::isize idx);
	axc::Range pyslice_to_range(axc::isize len, const nb::object &obj);
	std::vector<axc::Range> pyslices_to_ranges(const axr::Array &arr, const nb::object &obj);
	axc::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype);
	const std::string device_from_nb_device(int nb_device_id, int nb_device_type);
	nb::ndarray<nb::numpy> array_to_numpy(const axr::Array &arr);
	axr::ArrayPtr array_from_numpy(nb::ndarray<nb::numpy> &ndarr);
	axr::ArrayPtr array_from_torch(nb::ndarray<nb::pytorch> &ndarr);
	nb::ndarray<nb::pytorch> array_to_torch(const axr::Array &arr);
	axr::ArrayPtr full(const axc::ShapeView &view, const nb::object &obj, axc::DtypePtr dtype, const std::string &device_name = axd::default_device_name);
	axr::ArrayPtr full_like(axr::ArrayPtr other, const nb::object &obj, axc::DtypePtr dtype, const std::string &device_name = axd::default_device_name);
	axr::ArrayPtr slice(axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr transpose(axr::ArrayPtr arr, axc::isize start_dim, axc::isize end_dim);
	axr::ArrayPtr flatten(axr::ArrayPtr arr, axc::isize start_dim, axc::isize end_dim);
	axr::ArrayPtr squeeze(axr::ArrayPtr arr, axc::isize dim);
	axr::ArrayPtr unsqueeze(axr::ArrayPtr arr, axc::isize dim);
	axr::ArrayPtr pyobj_to_arr(const nb::object &obj, const std::string &device_name = axd::default_device_name);
}