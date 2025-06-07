#include "array.h"
#include "nn.h"
#include "optim.h"

NB_MODULE(arrayx, m) {
    auto m_core = m.def_submodule("core", "Core module");
    auto m_nn = m.def_submodule("nn", "Neural network module");
    auto m_optim = m.def_submodule("optim", "Optimizer module");

    // Dtype class and operations
    nb::enum_<axc::DtypeType>(m_core, "DtypeType")
        .value("BOOL", axc::DtypeType::BOOL)
        .value("INT", axc::DtypeType::INT)
        .value("FLOAT", axc::DtypeType::FLOAT);

    nb::class_<axc::Dtype>(m_core, "Dtype")
        .def_prop_ro("name", &axc::Dtype::get_name_str, "Get data type's name as string")
        .def_prop_ro("size", &axc::Dtype::get_size, "Get data type's size in bytes")
        .def_prop_ro("type", &axc::Dtype::get_type, "Get data type's type")
        .def("__str__", &axc::Dtype::str, "String representation of dtype");

    // Derived dtype classes
    nb::class_<axc::F32, axc::Dtype>(m_core, "F32", "32-bit floating point dtype");
    nb::class_<axc::I32, axc::Dtype>(m_core, "I32", "32-bit integer dtype");
    nb::class_<axc::Bool, axc::Dtype>(m_core, "Bool", "Boolean dtype");

    // Global dtype instances
    m_core.attr("f32") = &axc::f32;
    m_core.attr("i32") = &axc::i32;
    m_core.attr("b8") = &axc::b8;

    // Shape class
    nb::class_<axc::Shape>(m_core, "Shape")
        .def_prop_ro("offset", &axc::Shape::get_offset, "Get shape's offset")
        .def_prop_ro("view", &axc::Shape::get_view, "Get shape's view")
        .def_prop_ro("stride", &axc::Shape::get_stride, "Get shape's stride")
        .def_prop_ro("ndim", &axc::Shape::get_ndim, "Get shape's number of dimensions")
        .def_prop_ro("numel", &axc::Shape::get_numel, "Get shape's total number of elements")
        .def("__str__", &axc::Shape::str, "String representation of shape");

    // Device class
    nb::enum_<axd::DeviceType>(m_core, "DeviceType")
        .value("CPU", axd::DeviceType::CPU)
        .value("MPS", axd::DeviceType::MPS);

    nb::class_<axd::Device>(m_core, "Device")
        .def_prop_ro("type", &axd::Device::get_type, "Get device's type")
        .def_prop_ro("id", &axd::Device::get_id, "Get device's ID")
        .def_prop_ro("name", &axd::Device::get_name, "Get device's name")
        .def("__str__", &axd::Device::str, "String representation of device");

    // Backend class
    nb::class_<axr::Backend>(m_core, "Backend")
        .def_static("init", &axr::Backend::init, "Initialize backend")
        .def_static("cleanup", &axr::Backend::cleanup, "Shutdown backend");

    // Array class
    nb::class_<axr::Array>(m_core, "Array")
        // Properties
        .def_prop_ro("id", [](const axr::Array &arr) { return arr.get_id().str(); }, "Get array's ID")
        .def_prop_ro("shape", &axr::Array::get_shape, "Get array's shape")
        .def_prop_ro("dtype", &axr::Array::get_dtype, "Get array's data type")
        .def_prop_ro("device", &axr::Array::get_device, "Get array's device")
        .def_prop_ro("grad", &axr::Array::get_grad, "Get array's gradient")
        .def_prop_ro("ndim", &axr::Array::get_ndim, "Get array's number of dimensions")
        .def_prop_ro("numel", &axr::Array::get_numel, "Get array's total number of elements")
        .def_prop_ro("offset", &axr::Array::get_offset, "Get array's offset")
        .def_prop_ro("view", &axr::Array::get_view, "Get array's view")
        .def_prop_ro("stride", &axr::Array::get_stride, "Get array's stride")
        .def_prop_ro("ptr", &axr::Array::get_ptr, "Get array's raw data pointer")
        .def_prop_ro("itemsize", &axr::Array::get_itemsize, "Get array's element size in bytes")
        .def_prop_ro("nbytes", &axr::Array::get_nbytes, "Get array's total size in bytes")
        .def_prop_ro("is_contiguous", &axr::Array::is_contiguous, "Check if array is contiguous")
        .def_prop_rw("grad_enabled", &axr::Array::is_grad_enabled, &axr::Array::enable_grad, "enabled"_a = true, "Get/set array's gradient tracking")

        // N-dimensional array
        .def("numpy", &axb::array_to_numpy, nb::rv_policy::reference_internal, "Convert array to numpy array")
        .def_static("from_numpy", &axb::array_from_numpy, "array"_a, "Convert numpy array to array")
        .def("torch", &axb::array_to_torch, nb::rv_policy::reference_internal, "Convert array to Pytorch tensor")
        // .def_static("from_torch", &axb::array_from_torch, "tensor"_a, "Convert Pytorch tensor to array")
        .def("item", &axb::item, "Get array's only value")
        .def("graph", &axr::Array::graph_str, "Get array's computation graph representation")

        // Initializer operations
        .def_static("full", &axb::full, "view"_a, "c"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with specified value")
        .def_static("full_like", &axb::full_like, "other"_a, "c"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with specified value with same shape as the input array")
        .def_static("zeros", &axr::Array::zeros, "view"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with zeros")
        .def_static("ones", &axr::Array::ones, "view"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array filled with ones")
        .def_static("arange", &axr::Array::arange, "view"_a, "start"_a, "step"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array with evenly spaced values")
        .def_static("zeros_like", &axr::Array::zeros_like, "other"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array of zeros with same shape as input")
        .def_static("ones_like", &axr::Array::ones_like, "other"_a, "dtype"_a = &axc::f32, "device"_a = axd::default_device_name, "Create a new array of ones with same shape as input")

        // Element-wise operations
        .def("__add__", &axb::add, "rhs"_a, "Add two arrays element-wise")
        .def("__radd__", &axb::add, "rhs"_a, "Add two arrays element-wise")
        .def("__sub__", &axb::sub, "rhs"_a, "Subtract two arrays element-wise")
        .def("__rsub__", &axb::sub, "rhs"_a, "Subtract two arrays element-wise")
        .def("__mul__", &axb::mul, "rhs"_a, "Multiply two arrays element-wise")
        .def("__rmul__", &axb::mul, "rhs"_a, "Multiply two arrays element-wise")
        .def("__truediv__", &axb::div, "rhs"_a, "Divide two arrays element-wise")
        .def("__rtruediv__", &axb::div, "rhs"_a, "Divide two arrays element-wise")
        .def("__iadd__", &axb::self_add, "rhs"_a, "In-place add two arrays element-wise")
        .def("__isub__", &axb::self_sub, "rhs"_a, "In-place subtract two arrays element-wise")
        .def("__imul__", &axb::self_mul, "rhs"_a, "In-place multiply two arrays element-wise")
        .def("__itruediv__", &axb::self_div, "rhs"_a, "In-place divide two arrays element-wise")
        .def("__matmul__", &axr::Array::matmul, "rhs"_a, "Matrix multiply two arrays")
        .def("detach", &axr::Array::detach, "Detach array from computation graph")
        .def("exp", &axr::Array::exp, "in_place"_a = false, "Compute exponential of array elements")
        .def("log", &axr::Array::log, "in_place"_a = false, "Compute natural logarithm of array elements")
        .def("sqrt", &axr::Array::sqrt, "in_place"_a = false, "Compute square root of array elements")
        .def("sq", &axr::Array::sq, "in_place"_a = false, "Compute square of array elements")
        .def("neg", &axr::Array::neg, "in_place"_a = false, "Compute negative of array elements")
        .def("__neg__", &axb::neg, "Compute negative of array elements")
        .def("recip", &axr::Array::recip, "in_place"_a = false, "Compute reciprocal of array elements")

        // Comparison operations
        .def("__eq__", &axb::eq, "rhs"_a, "Element-wise equality comparison")
        .def("__ne__", &axb::neq, "rhs"_a, "Element-wise inequality comparison")
        .def("__lt__", &axb::lt, "rhs"_a, "Element-wise less than comparison")
        .def("__gt__", &axb::gt, "rhs"_a, "Element-wise greater than comparison")
        .def("__le__", &axb::leq, "rhs"_a, "Element-wise less than or equal comparison")
        .def("__ge__", &axb::geq, "rhs"_a, "Element-wise greater than or equal comparison")
        .def("minimum", &axb::minimum, "rhs"_a, "Element-wise minimum comparison")
        .def("maximum", &axb::maximum, "rhs"_a, "Element-wise maximum comparison")

        // Reduction operations
        .def("sum", &axb::sum, "dims"_a = axc::ShapeDims{}, "Sum array elements along specified dimensions")
        .def("mean", &axb::mean, "dims"_a = axc::ShapeDims{}, "Mean value along specified dimensions")
        .def("max", &axb::max, "dims"_a = axc::ShapeDims{}, "Maximum value along specified dimensions")
        .def("min", &axb::min, "dims"_a = axc::ShapeDims{}, "Minimum value along specified dimensions")
        .def("argmax", &axb::argmax, "dims"_a = axc::ShapeDims{}, "Indices of maximum values along specified dimensions")
        .def("argmin", &axb::argmin, "dims"_a = axc::ShapeDims{}, "Indices of minimum values along specified dimensions")

        // Shape operations
        .def("broadcast", &axr::Array::broadcast, "view"_a, "Broadcast array to new shape")
        .def("broadcast_to", &axr::Array::broadcast_to, "view"_a, "Broadcast array to target shape")
        .def("__getitem__", &axb::slice, "index"_a, "Slice array along specified dimensions")
        .def("reshape", &axr::Array::reshape, "view"_a, "Reshape array to new dimensions")
        .def("flatten", &axb::flatten, "start_dim"_a = 0, "end_dim"_a = -1, "Flatten dimensions from start to end")
        .def("squeeze", &axb::squeeze, "dims"_a = axc::ShapeDims{}, "Remove single-dimensional entry from array")
        .def("unsqueeze", &axb::unsqueeze, "dims"_a = axc::ShapeDims{}, "Add single-dimensional entry to array")
        .def("permute", &axb::permute, "dims"_a, "Permute array dimensions")
        .def("transpose", &axb::transpose, "start_dim"_a = -2, "end_dim"_a = -1, "Transpose array dimensions")

        // Type operations
        .def("astype", &axr::Array::astype, "dtype"_a, "Cast array to specified dtype")

        // Evaluation and backward
        .def("eval", &axr::Array::eval, "Evaluate array and materialize values")
        .def("backward", &axr::Array::backward, "Compute gradients through backpropagation")

        // String representation
        .def("__str__", &axr::Array::str, "String representation of array");

    nb::class_<axnn::Module, axb::PyModule>(m_nn, "Module")
        .def(nb::init<>())
        .def("__call__", &axnn::Module::operator(), "x"_a, "Call the nn module using the forward hook")
        .def("forward", &axnn::Module::forward, "x"_a, "Forward the nn module, can be overidden")
        .def("parameters", &axnn::Module::parameters, "Get the parameters of the nn module, can be overidden")
        .def("jit", &axnn::Module::jit, "x"_a, "JIT-compile the nn module");

    nb::class_<axnn::CrossEntropyLoss, axnn::Module>(m_nn, "CrossEntropyLoss")
        .def(nb::init<>());

    nb::class_<axo::Optimizer, axb::PyOptimizer>(m_optim, "Optimizer")
        .def(nb::init<const axr::ArrayVec &, float>(), "params"_a, "lr"_a = 1e-3, "Base optimizer")
        .def("forward", &axo::Optimizer::forward, "Parameter update function")
        .def("step", &axo::Optimizer::step, "Update module parameters");

    nb::class_<axo::GradientDescent, axo::Optimizer>(m_optim, "GradientDescent")
        .def(nb::init<const axr::ArrayVec &, float>(), "params"_a, "lr"_a = 1e-3, "Gradient Descent optimizer");

    m_nn.def("linear", &axnn::linear, "x"_a, "weight"_a, "Functional linear without bias");
    m_nn.def("linear_with_bias", &axnn::linear_with_bias, "x"_a, "weight"_a, "bias"_a, "Functional linear with bias");
    m_nn.def("relu", &axnn::relu, "x"_a, "ReLU activation function");
    m_nn.def("onehot", &axnn::onehot, "x"_a, "num_classes"_a = -1, "One-hot encode input array");
    m_nn.def("cross_entropy_loss", &axnn::cross_entropy_loss, "x"_a, "y"_a, "Compute cross-entropy loss between input x and target y");
}
