#include "array.h"

namespace ax::array
{
	void Array::eval()
	{
		if (compute_graph == nullptr)
		{
			compute_graph = std::make_shared<ComputeGraph>(op);
		}
		compute_graph->forward();
		get_backend_runner()->forward(compute_graph);
	}

	void Array::backward()
	{
		compute_graph->backward();
		get_backend_runner()->backward(compute_graph);
	}

	ArrayPtr Array::arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::arange(view, start, step, dtype, device);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::zeros(const ShapeView &view, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::zeros(view, dtype, device);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::ones(const ShapeView &view, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::ones(view, dtype, device);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::zeros_like(ArrayPtr other, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::zeros_like(other->op, dtype, device);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::ones_like(ArrayPtr other, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::ones_like(other->op, dtype, device);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::from_ptr(ptr, nbytes, shape, dtype, device);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::add(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::add(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::sub(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::sub(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::mul(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::mul(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::div(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::div(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::self_add(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::self_add(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::self_sub(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::self_sub(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::self_mul(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::self_mul(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::self_div(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::self_div(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::matmul(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::matmul(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::exp(bool in_place) const
	{
		OpPtr out_op = ax::graph::exp(op, in_place);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::log(bool in_place) const
	{
		OpPtr out_op = ax::graph::log(op, in_place);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::sqrt(bool in_place) const
	{
		OpPtr out_op = ax::graph::sqrt(op, in_place);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::sq(bool in_place) const
	{
		OpPtr out_op = ax::graph::sq(op, in_place);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::neg(bool in_place) const
	{
		OpPtr out_op = ax::graph::neg(op, in_place);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::recip(bool in_place) const
	{
		OpPtr out_op = ax::graph::recip(op, in_place);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::eq(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::eq(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::neq(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::neq(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::lt(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::lt(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::gt(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::gt(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::leq(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::leq(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::geq(ArrayPtr rhs) const
	{
		OpPtr out_op = ax::graph::geq(op, rhs->op);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::sum(const ShapeDims &dims) const
	{
		OpPtr out_op = ax::graph::sum(op, dims);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::max(const ShapeDims &dims) const
	{
		OpPtr out_op = ax::graph::max(op, dims);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::min(const ShapeDims &dims) const
	{
		OpPtr out_op = ax::graph::min(op, dims);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::argmax(const ShapeDims &dims) const
	{
		OpPtr out_op = ax::graph::argmax(op, dims);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::argmin(const ShapeDims &dims) const
	{
		OpPtr out_op = ax::graph::argmin(op, dims);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::broadcast(const ShapeView &view) const
	{
		OpPtr out_op = ax::graph::broadcast(op, view);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::broadcast_to(const ShapeView &view) const
	{
		OpPtr out_op = ax::graph::broadcast_to(op, view);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::slice(const RangeVec &ranges) const
	{
		OpPtr out_op = ax::graph::slice(op, ranges);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::reshape(const ShapeView &view) const
	{
		OpPtr out_op = ax::graph::reshape(op, view);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::flatten(isize start_dim, isize end_dim) const
	{
		OpPtr out_op = ax::graph::flatten(op, start_dim, end_dim);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::squeeze(isize dim) const
	{
		OpPtr out_op = ax::graph::squeeze(op, dim);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::unsqueeze(isize dim) const
	{
		OpPtr out_op = ax::graph::unsqueeze(op, dim);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::permute(const ShapeDims &dims) const
	{
		OpPtr out_op = ax::graph::permute(op, dims);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::transpose(isize start_dim, isize end_dim) const
	{
		OpPtr out_op = ax::graph::transpose(op, start_dim, end_dim);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::astype(DtypePtr dtype) const
	{
		OpPtr out_op = ax::graph::astype(op, dtype);
		return std::make_shared<Array>(out_op);
	}
}
