#include "array.h"

NB_MODULE(arrayx, m)
{
	// Dtype class and operations
	nb::class_<axc::Dtype>(m, "Dtype")
		.def("name", &axc::Dtype::get_name_str, "Get dtype name as string")
		.def("size", &axc::Dtype::get_size, "Get size in bytes")
		.def("__str__", &axc::Dtype::str, "String representation of dtype");

	// Derived dtype classes
	nb::class_<axc::F32, axc::Dtype>(m, "F32", "32-bit floating point dtype");
	nb::class_<axc::I32, axc::Dtype>(m, "I32", "32-bit integer dtype");
	nb::class_<axc::Bool, axc::Dtype>(m, "Bool", "Boolean dtype");

	// Global dtype instances
	m.attr("f32") = &axc::f32;
	m.attr("i32") = &axc::i32;
	m.attr("bool") = &axc::b8;

	// Shape class
	nb::class_<axc::Shape>(m, "Shape")
		.def_prop_ro("offset", &axc::Shape::get_offset, "Get shape offset")
		.def_prop_ro("view", &axc::Shape::get_view, "Get shape view")
		.def_prop_ro("stride", &axc::Shape::get_stride, "Get shape stride")
		.def_prop_ro("ndim", &axc::Shape::get_ndim, "Get number of dimensions")
		.def_prop_ro("numel", &axc::Shape::get_numel, "Get total number of elements")
		.def("__str__", &axc::Shape::str, "String representation of shape");

	// Device class
	nb::enum_<axd::DeviceType>(m, "DeviceType")
		.value("CPU", axd::DeviceType::CPU)
		.value("MPS", axd::DeviceType::MPS);

	nb::class_<axd::Device>(m, "Device")
		.def("type", &axd::Device::get_type, "Get device type")
		.def("id", &axd::Device::get_id, "Get device ID")
		.def("name", &axd::Device::get_name, "Get device name")
		.def("__str__", &axd::Device::str, "String representation of device");

	// Backend class
	nb::class_<axr::Backend>(m, "Backend")
		.def_static("init", &axr::Backend::init, "Initialize backend")
		.def_static("cleanup", &axr::Backend::cleanup, "Shutdown backend");

	// Array class
	nb::class_<axr::Array>(m, "Array")
		// Properties
		.def_prop_ro("shape", &axr::Array::get_shape, "Get array shape")
		.def_prop_ro("dtype", &axr::Array::get_dtype, "Get array data type")
		.def_prop_ro("device", &axr::Array::get_device, "Get array device")
		.def_prop_ro("grad", &axr::Array::get_grad, "Get array gradient")
		.def_prop_ro("ndim", &axr::Array::get_ndim, "Get number of dimensions")
		.def_prop_ro("numel", &axr::Array::get_numel, "Get total number of elements")
		.def_prop_ro("offset", &axr::Array::get_offset, "Get array offset")
		.def_prop_ro("view", &axr::Array::get_view, "Get array view")
		.def_prop_ro("stride", &axr::Array::get_stride, "Get array stride")
		.def_prop_ro("ptr", &axr::Array::get_ptr, "Get raw data pointer")
		.def_prop_ro("itemsize", &axr::Array::get_itemsize, "Get size of each element in bytes")
		.def_prop_ro("nbytes", &axr::Array::get_nbytes, "Get total size in bytes")
		.def_prop_ro("is_contiguous", &axr::Array::is_contiguous, "Check if array is contiguous")

		// N-dimensional array
		.def("numpy", &axb::array_to_numpy, nb::rv_policy::reference_internal, "Convert array to numpy array")
		.def_static("from_numpy", &axb::array_from_numpy, "array"_a, "Convert numpy array to array")
		.def("torch", &axb::array_to_torch, nb::rv_policy::reference_internal, "Convert array to Pytorch tensor")
		.def_static("from_torch", &axb::array_from_torch, "tensor"_a, "Convert Pytorch tensor to array")

		// Initializer operations
		.def_static("full", &axb::full, "view"_a, "c"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with specified value")
		.def_static("full_like", &axb::full_like, "other"_a, "c"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with specified value with same shape as the input array")
		.def_static("zeros", &axr::Array::zeros, "view"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with zeros")
		.def_static("ones", &axr::Array::ones, "view"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with ones")
		.def_static("arange", &axr::Array::arange, "view"_a, "start"_a, "step"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array with evenly spaced values")
		.def_static("zeros_like", &axr::Array::zeros_like, "other"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array of zeros with same shape as input")
		.def_static("ones_like", &axr::Array::ones_like, "other"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array of ones with same shape as input")

		// Element-wise operations
		.def("__add__", &axr::Array::add, "rhs"_a, "Add two arrays element-wise")
		.def("__sub__", &axr::Array::sub, "rhs"_a, "Subtract two arrays element-wise")
		.def("__mul__", &axr::Array::mul, "rhs"_a, "Multiply two arrays element-wise")
		.def("__truediv__", &axr::Array::div, "rhs"_a, "Divide two arrays element-wise")
		.def("__iadd__", &axr::Array::self_add, "rhs"_a, "In-place add two arrays element-wise")
		.def("__isub__", &axr::Array::self_sub, "rhs"_a, "In-place subtract two arrays element-wise")
		.def("__imul__", &axr::Array::self_mul, "rhs"_a, "In-place multiply two arrays element-wise")
		.def("__itruediv__", &axr::Array::self_div, "rhs"_a, "In-place divide two arrays element-wise")
		.def("__matmul__", &axr::Array::matmul, "rhs"_a, "Matrix multiply two arrays")
		.def("exp", &axr::Array::exp, "in_place"_a = false, "Compute exponential of array elements")
		.def("log", &axr::Array::log, "in_place"_a = false, "Compute natural logarithm of array elements")
		.def("sqrt", &axr::Array::sqrt, "in_place"_a = false, "Compute square root of array elements")
		.def("sq", &axr::Array::sq, "in_place"_a = false, "Compute square of array elements")
		.def("neg", &axr::Array::neg, "in_place"_a = false, "Compute negative of array elements")
		.def("__neg__", &axr::Array::neg, "Compute negative of array elements")
		.def("recip", &axr::Array::recip, "in_place"_a = false, "Compute reciprocal of array elements")

		// Comparison operations
		.def("__eq__", &axr::Array::eq, "rhs"_a, "Element-wise equality comparison")
		.def("__ne__", &axr::Array::neq, "rhs"_a, "Element-wise inequality comparison")
		.def("__lt__", &axr::Array::lt, "rhs"_a, "Element-wise less than comparison")
		.def("__gt__", &axr::Array::gt, "rhs"_a, "Element-wise greater than comparison")
		.def("__le__", &axr::Array::leq, "rhs"_a, "Element-wise less than or equal comparison")
		.def("__ge__", &axr::Array::geq, "rhs"_a, "Element-wise greater than or equal comparison")

		// Reduction operations
		.def("sum", &axr::Array::sum, "dims"_a = axc::ShapeDims{}, "Sum array elements along specified dimensions")
		.def("max", &axr::Array::max, "dims"_a = axc::ShapeDims{}, "Maximum value along specified dimensions")
		.def("min", &axr::Array::min, "dims"_a = axc::ShapeDims{}, "Minimum value along specified dimensions")
		.def("argmax", &axr::Array::argmax, "dims"_a = axc::ShapeDims{}, "Indices of maximum values along specified dimensions")
		.def("argmin", &axr::Array::argmin, "dims"_a = axc::ShapeDims{}, "Indices of minimum values along specified dimensions")

		// Shape operations
		.def("broadcast", &axr::Array::broadcast, "view"_a, "Broadcast array to new shape")
		.def("broadcast_to", &axr::Array::broadcast_to, "view"_a, "Broadcast array to target shape")
		.def("__getitem__", &axb::slice, "index"_a, "Slice array along specified dimensions")
		.def("reshape", &axr::Array::reshape, "view"_a, "Reshape array to new dimensions")
		.def("flatten", &axb::flatten, "start_dim"_a, "end_dim"_a, "Flatten dimensions from start to end")
		.def("squeeze", &axb::squeeze, "dim"_a, "Remove single-dimensional entry from array")
		.def("unsqueeze", &axb::unsqueeze, "dim"_a, "Add single-dimensional entry to array")
		.def("permute", &axr::Array::permute, "dims"_a, "Permute array dimensions")
		.def("transpose", &axb::transpose, "start_dim"_a = 0, "end_dim"_a = 1, "Transpose array dimensions")

		// Type operations
		.def("astype", &axr::Array::astype, "dtype"_a, "Cast array to specified dtype")

		// Evaluation and backward
		.def("eval", &axr::Array::eval, "Evaluate array and materialize values")
		.def("backward", &axr::Array::backward, "Compute gradients through backpropagation")

		// String representation
		.def("__str__", &axr::Array::str, "String representation of array");
}
