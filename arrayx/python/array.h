#pragma once

#include "bind.h"

namespace ax::bind
{
	template <class T>
	nb::ndarray<nb::numpy> array_to_numpy_impl(axr::Array &arr)
	{
		arr.eval();
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
	nb::ndarray<nb::pytorch> array_to_torch_impl(axr::Array &arr)
	{
		arr.eval();
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
			device = nb::device::cpu::value;
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

	template <class OpFunc>
	axr::ArrayPtr binary(const axr::Array &arr, const nb::object &obj, OpFunc op_func)
	{
		if (nb::isinstance<axr::Array>(obj))
		{
			return op_func(arr, nb::cast<axr::ArrayPtr>(obj));
		}
		else if (nb::isinstance<nb::float_>(obj))
		{
			return op_func(arr, nb::cast<float>(obj));
		}
		else if (nb::isinstance<nb::int_>(obj))
		{
			return op_func(arr, nb::cast<int>(obj));
		}
		else if (nb::isinstance<nb::bool_>(obj))
		{
			return op_func(arr, nb::cast<bool>(obj));
		}
		throw axc::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
	}

	axc::isize get_pyindex(axc::isize len, axc::isize idx);
	axc::ShapeDims get_pyindices(axc::isize len, const axc::ShapeDims &dims);
	axc::Range pyslice_to_range(axc::isize len, const nb::object &obj);
	std::vector<axc::Range> pyslices_to_ranges(const axr::Array &arr, const nb::object &obj);
	axc::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype);
	const std::string device_from_nb_device(int nb_device_id, int nb_device_type);
	nb::ndarray<nb::numpy> array_to_numpy(axr::Array &arr);
	axr::ArrayPtr array_from_numpy(nb::ndarray<nb::numpy> &ndarr);
	axr::ArrayPtr array_from_torch(nb::ndarray<nb::pytorch> &ndarr);
	nb::ndarray<nb::pytorch> array_to_torch(axr::Array &arr);
	nb::object item(axr::Array &arr);
	axr::ArrayPtr full(const axc::ShapeView &view, const nb::object &obj, axc::DtypePtr dtype, const std::string &device_name = axd::default_device_name);
	axr::ArrayPtr full_like(axr::ArrayPtr other, const nb::object &obj, axc::DtypePtr dtype, const std::string &device_name = axd::default_device_name);
	axr::ArrayPtr neg(const axr::Array &arr);
	axr::ArrayPtr add(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr self_add(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr sub(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr self_sub(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr mul(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr self_mul(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr div(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr self_div(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr eq(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr neq(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr lt(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr gt(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr leq(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr geq(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr slice(const axr::Array &arr, const nb::object &obj);
	axr::ArrayPtr permute(const axr::Array &arr, const ax::core::ShapeDims &dims);
	axr::ArrayPtr transpose(const axr::Array &arr, axc::isize start_dim, axc::isize end_dim);
	axr::ArrayPtr flatten(const axr::Array &arr, axc::isize start_dim, axc::isize end_dim);
	axr::ArrayPtr squeeze(const axr::Array &arr, const axc::ShapeDims &dims = {});
	axr::ArrayPtr unsqueeze(const axr::Array &arr, const axc::ShapeDims &dims = {});
	axr::ArrayPtr sum(const axr::Array &arr, const axc::ShapeDims &dims);
	axr::ArrayPtr mean(const axr::Array &arr, const axc::ShapeDims &dims);
	axr::ArrayPtr max(const axr::Array &arr, const axc::ShapeDims &dims);
	axr::ArrayPtr min(const axr::Array &arr, const axc::ShapeDims &dims);
	axr::ArrayPtr argmax(const axr::Array &arr, const axc::ShapeDims &dims);
	axr::ArrayPtr argmin(const axr::Array &arr, const axc::ShapeDims &dims);
}