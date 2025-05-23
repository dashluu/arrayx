#include "array.h"

namespace ax
{
	ArrayPtr Array::full(const ShapeView &view, int c, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::full(view, c, dtype, device);
		return std::make_shared<Array>(out_op);
	}

	ArrayPtr Array::arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, const std::string &device_name)
	{
		DevicePtr device = get_backend_device(device_name);
		OpPtr out_op = ax::graph::arange(view, start, step, dtype, device);
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
}
