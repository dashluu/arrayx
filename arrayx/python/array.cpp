#include "array.h"

namespace ax::bind
{
	axc::isize get_pyindex(axc::isize len, axc::isize idx)
	{
		if (idx < -len || idx >= len)
		{
			throw axc::OutOfRange(idx, -len, len);
		}
		return idx < 0 ? idx + len : idx;
	}

	axc::ShapeDims get_pyindices(axc::isize len, axc::ShapeDims &dims)
	{
		if (!dims.empty())
		{
			std::transform(dims.begin(), dims.end(), dims.begin(), [len](auto dim)
						   { return get_pyindex(len, dim); });
		}
		return dims;
	}

	axc::Range pyslice_to_range(axc::isize len, const nb::object &obj)
	{
		// Note: no need to check for out-of-bounds indices when converting to range
		// Shape does the checking eventually
		if (!nb::isinstance<nb::slice>(obj))
		{
			throw axc::NanobindInvalidArgumentType(get_pyclass(obj), "slice");
		}
		auto slice = nb::cast<nb::slice>(obj);
		bool start_none = slice.attr("start").is_none();
		bool stop_none = slice.attr("stop").is_none();
		bool step_none = slice.attr("step").is_none();
		axc::isize start, stop, step;
		if (step_none)
		{
			start = start_none ? 0 : get_pyindex(len, nb::cast<axc::isize>(slice.attr("start")));
			stop = stop_none ? len : get_pyindex(len, nb::cast<axc::isize>(slice.attr("stop")));
			return axc::Range(start, stop, 1);
		}
		step = nb::cast<axc::isize>(slice.attr("step"));
		if (step > 0)
		{
			start = start_none ? 0 : get_pyindex(len, nb::cast<axc::isize>(slice.attr("start")));
			stop = stop_none ? len : get_pyindex(len, nb::cast<axc::isize>(slice.attr("stop")));
		}
		else
		{
			start = start_none ? len - 1 : get_pyindex(len, nb::cast<axc::isize>(slice.attr("start")));
			stop = stop_none ? -1 : get_pyindex(len, nb::cast<axc::isize>(slice.attr("stop")));
		}
		return axc::Range(start, stop, step);
	}

	std::vector<axc::Range> pyslices_to_ranges(const axr::Array &arr, const nb::object &obj)
	{
		std::vector<axc::Range> ranges;
		const axc::Shape &shape = arr.get_shape();
		// obj can be an int, a slice, or a sequence of ints or slices
		if (nb::isinstance<nb::int_>(obj))
		{
			axc::isize idx = get_pyindex(shape[0], nb::cast<axc::isize>(obj));
			ranges.emplace_back(idx, idx + 1, 1);
			for (axc::isize i = 1; i < shape.get_ndim(); i++)
			{
				ranges.emplace_back(0, shape[i], 1);
			}
			return ranges;
		}
		else if (nb::isinstance<nb::slice>(obj))
		{
			ranges.push_back(pyslice_to_range(shape[0], obj));
			for (axc::isize i = 1; i < shape.get_ndim(); i++)
			{
				ranges.emplace_back(0, shape[i], 1);
			}
			return ranges;
		}
		else if (nb::isinstance<nb::sequence>(obj) && !nb::isinstance<nb::str>(obj))
		{
			// Object is a sequence but not a string
			auto sequence = nb::cast<nb::sequence>(obj);
			size_t seq_len = nb::len(sequence);
			if (seq_len > shape.get_ndim())
			{
				throw axc::OutOfRange(seq_len, 1, shape.get_ndim() + 1);
			}
			for (size_t i = 0; i < seq_len; i++)
			{
				auto elm = sequence[i];
				// elm must be a sequence of ints or slices
				if (nb::isinstance<nb::int_>(elm))
				{
					axc::isize idx = get_pyindex(shape[i], nb::cast<axc::isize>(elm));
					ranges.emplace_back(idx, idx + 1, 1);
				}
				else if (nb::isinstance<nb::slice>(elm))
				{
					ranges.push_back(pyslice_to_range(shape[i], elm));
				}
				else
				{
					throw axc::NanobindInvalidArgumentType(get_pyclass(elm), "int, slice");
				}
			}
			for (axc::isize i = seq_len; i < shape.get_ndim(); i++)
			{
				ranges.emplace_back(0, shape[i], 1);
			}
			return ranges;
		}
		throw axc::NanobindInvalidArgumentType(get_pyclass(obj), "int, slice, sequence");
	}

	axc::DtypePtr dtype_from_nb_dtype(nb::dlpack::dtype nb_dtype)
	{
		if (nb_dtype == nb::dtype<float>())
		{
			return &axc::f32;
		}
		else if (nb_dtype == nb::dtype<int>())
		{
			return &axc::i32;
		}
		else if (nb_dtype == nb::dtype<bool>())
		{
			return &axc::b8;
		}
		throw nb::type_error("Nanobind data type cannot be converted to arrayx data type.");
	}

	const std::string device_from_nb_device(int nb_device_id, int nb_device_type)
	{
		if (nb_device_type == nb::device::cpu::value)
		{
			return "cpu:" + std::to_string(nb_device_id);
		}
		else if (nb_device_type == nb::device::metal::value)
		{
			return "mps:" + std::to_string(nb_device_id);
		}
		throw axc::UnsupportedNanobindDevice();
	}

	nb::ndarray<nb::numpy> array_to_numpy(axr::Array &arr)
	{
		switch (arr.get_dtype()->get_name())
		{
		case axc::DtypeName::F32:
			return array_to_numpy_impl<float>(arr);
		case axc::DtypeName::I32:
			return array_to_numpy_impl<int>(arr);
		default:
			return array_to_numpy_impl<bool>(arr);
		}
	}

	axr::Array array_from_numpy(nb::ndarray<nb::numpy> &ndarr)
	{
		axc::ShapeView view;
		axc::ShapeStride stride;

		for (size_t i = 0; i < ndarr.ndim(); ++i)
		{
			view.push_back(ndarr.shape(i));
			stride.push_back(ndarr.stride(i));
		}

		axc::Shape shape(0, view, stride);
		uint8_t *ptr = reinterpret_cast<uint8_t *>(ndarr.data());
		axc::DtypePtr dtype = dtype_from_nb_dtype(ndarr.dtype());
		return axr::Array::from_ptr(ptr, ndarr.nbytes(), shape, dtype, axd::default_device_name);
	}

	axr::Array array_from_torch(nb::ndarray<nb::pytorch> &ndarr)
	{
		axc::ShapeView view;
		axc::ShapeStride stride;

		for (size_t i = 0; i < ndarr.ndim(); ++i)
		{
			view.push_back(ndarr.shape(i));
			stride.push_back(ndarr.stride(i));
		}

		axc::Shape shape(0, view, stride);
		uint8_t *ptr = reinterpret_cast<uint8_t *>(ndarr.data());
		axc::DtypePtr dtype = dtype_from_nb_dtype(ndarr.dtype());
		const std::string &device_name = device_from_nb_device(ndarr.device_id(), ndarr.device_type());
		return axr::Array::from_ptr(ptr, ndarr.nbytes(), shape, dtype, device_name);
	}

	nb::ndarray<nb::pytorch> array_to_torch(axr::Array &arr)
	{
		switch (arr.get_dtype()->get_name())
		{
		case axc::DtypeName::F32:
			return array_to_torch_impl<float>(arr);
		case axc::DtypeName::I32:
			return array_to_torch_impl<int>(arr);
		default:
			return array_to_torch_impl<bool>(arr);
		}
	}

	nb::object item(axr::Array &arr)
	{
		axc::isize value = arr.item();
		axc::DtypePtr dtype = arr.get_dtype();

		switch (dtype->get_name())
		{
		case axc::DtypeName::F32:
			return nb::cast<float>(std::bit_cast<float>(static_cast<int32_t>(value)));
		case axc::DtypeName::I32:
			return nb::cast<int>(value);
		default:
			return nb::cast<bool>(value);
		}
	}

	axr::Array full(const axc::ShapeView &view, const nb::object &obj, axc::DtypePtr dtype, const std::string &device_name)
	{
		if (nb::isinstance<nb::float_>(obj))
		{
			return axr::Array::full(view, nb::cast<float>(obj), dtype, device_name);
		}
		else if (nb::isinstance<nb::int_>(obj))
		{
			return axr::Array::full(view, nb::cast<int>(obj), dtype, device_name);
		}
		else if (nb::isinstance<nb::bool_>(obj))
		{
			return axr::Array::full(view, nb::cast<bool>(obj), dtype, device_name);
		}
		throw axc::NanobindInvalidArgumentType(get_pyclass(obj), "float, int, bool, Array");
	}

	axr::Array full_like(const axr::Array &other, const nb::object &obj, axc::DtypePtr dtype, const std::string &device_name)
	{
		return full(other.get_view(), obj, dtype, device_name);
	}

	axr::Array neg(const axr::Array &arr)
	{
		return arr.neg();
	}

	axr::Array add(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a + b; });
	}

	axr::Array self_add(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a += b; });
	}

	axr::Array sub(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a - b; });
	}

	axr::Array self_sub(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a -= b; });
	}

	axr::Array mul(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a * b; });
	}

	axr::Array self_mul(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a *= b; });
	}

	axr::Array div(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a / b; });
	}

	axr::Array self_div(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a /= b; });
	}

	axr::Array eq(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a == b; });
	}

	axr::Array neq(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a != b; });
	}

	axr::Array lt(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a < b; });
	}

	axr::Array gt(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a > b; });
	}

	axr::Array leq(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a <= b; });
	}

	axr::Array geq(const axr::Array &arr, const nb::object &obj)
	{
		return binary(arr, obj, [](const auto &a, const auto &b)
					  { return a >= b; });
	}

	axr::Array slice(const axr::Array &arr, const nb::object &obj)
	{
		return arr.slice(axb::pyslices_to_ranges(arr, obj));
	}

	axr::Array permute(const axr::Array &arr, ax::core::ShapeDims &dims)
	{
		return arr.permute(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array transpose(const axr::Array &arr, axc::isize start_dim, axc::isize end_dim)
	{
		return arr.transpose(get_pyindex(arr.get_shape().get_ndim(), start_dim), get_pyindex(arr.get_shape().get_ndim(), end_dim));
	}

	axr::Array flatten(const axr::Array &arr, axc::isize start_dim, axc::isize end_dim)
	{
		return arr.flatten(get_pyindex(arr.get_shape().get_ndim(), start_dim), get_pyindex(arr.get_shape().get_ndim(), end_dim));
	}

	axr::Array squeeze(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.squeeze(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array unsqueeze(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.unsqueeze(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array sum(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.sum(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array mean(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.mean(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array max(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.max(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array min(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.min(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array argmax(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.argmax(get_pyindices(arr.get_shape().get_ndim(), dims));
	}

	axr::Array argmin(const axr::Array &arr, axc::ShapeDims &dims)
	{
		return arr.argmin(get_pyindices(arr.get_shape().get_ndim(), dims));
	}
}