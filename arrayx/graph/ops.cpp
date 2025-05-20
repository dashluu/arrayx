#include "ops.h"

namespace ax::graph
{
	void Op::init_grad(bool with_zeros)
	{
		if (grad == nullptr)
		{
			DtypePtr dtype = output->get_dtype();
			if (dtype->get_type() != DtypeType::FLOAT)
			{
				throw std::runtime_error("Only arrays of floating-point types can have gradients but array " + output->get_id().str() + " has type " + dtype->str());
			}
			const ShapeView &view = output->get_shape().get_view();
			DtypePtr grad_dtype = float_dtype_by_dtype.at(dtype);
			DevicePtr device = output->get_device();
			grad = with_zeros ? zeros(view, grad_dtype, device) : ones(view, grad_dtype, device);
		}
	}

	void Op::update_grad(OpPtr grad, bool sub)
	{
		this->grad = sub ? self_sub(this->grad, grad) : self_add(this->grad, grad);
		this->gradroot = this->grad;
	}

	void AddOp::backward() const
	{
		// In-place or not, gradient should be computed properly
		// z = x + y
		// dx += dz
		// dy += dz
		lhs->init_grad();
		lhs->update_grad(grad);
		rhs->init_grad();
		rhs->update_grad(grad);
	}

	void SubOp::backward() const
	{
		// z = x + y
		// dx += dz
		// dy -= dz
		lhs->init_grad();
		lhs->update_grad(grad);
		rhs->init_grad();
		rhs->update_grad(grad, true);
	}

	void MulOp::backward() const
	{
		// z = x*y
		// dx += dz*y
		// dy += dz*x
		// Use detach to prevent circular dependencies
		lhs->init_grad();
		lhs->update_grad(mul(grad, detach(rhs)));
		rhs->init_grad();
		rhs->update_grad(mul(grad, detach(lhs)));
	}

	void DivOp::backward() const
	{
		// z = x/y
		// dx += dz * (1/y)
		// dy += dz * (-x / y**2)
		// dy -= dz * (z / y)
		// Use detach to prevent circular dependencies
		lhs->init_grad();
		lhs->update_grad(div(grad, detach(rhs)));
		rhs->init_grad();
		OpPtr detached_this = detach(std::const_pointer_cast<Op>(shared_from_this()));
		rhs->update_grad(mul(grad, div(detached_this, detach(rhs))), true);
	}

	const std::string UnaryOp::str() const
	{
		return Op::str() + ", in-place: " + std::to_string(in_place) + ", operand: " + operand->get_output()->get_id().str();
	}

	const std::string BinaryOp::str() const
	{
		return Op::str() + ", in-place: " + std::to_string(in_place) + ", lhs: " + lhs->get_output()->get_id().str() + ", rhs: " + rhs->get_output()->get_id().str();
	}

	const std::string MatmulOp::str() const
	{
		return Op::str() + ", lhs: " + lhs->get_output()->get_id().str() + ", rhs: " + rhs->get_output()->get_id().str();
	}

	const std::string TransformOp::str() const
	{
		return Op::str() + ", operand: " + operand->get_output()->get_id().str();
	}

	const std::string ReduceOp::str() const
	{
		return Op::str() + ", operand: " + operand->get_output()->get_id().str() + ", dims: " + vnumstr(dims) + ", default value: " + std::to_string(default_val);
	}

	OpPtr detach(OpPtr op)
	{
		return std::make_shared<NoopOp>(op->get_output());
	}

	OpPtr full(const ShapeView &view, int c, DtypePtr dtype, DevicePtr device)
	{
		ArrayPtr arr = Array::empty(Shape(view), dtype, device);
		OpPtr op = std::make_shared<FullOp>(arr, view, c, dtype);
		return op;
	}

	OpPtr zeros(const ShapeView &view, DtypePtr dtype, DevicePtr device)
	{
		return full(view, dtype->zero(), dtype, device);
	}

	OpPtr ones(const ShapeView &view, DtypePtr dtype, DevicePtr device)
	{
		return full(view, dtype->one(), dtype, device);
	}

	OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, DevicePtr device)
	{
		ArrayPtr arr = Array::empty(Shape(view), dtype, device);
		OpPtr op = std::make_shared<ArangeOp>(arr, view, start, step, dtype);
		return op;
	}

	OpPtr from_buff(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device)
	{
		ArrayPtr arr = Array::from_ptr(ptr, nbytes, shape, dtype, device);
		OpPtr op = std::make_shared<BuffOp>(arr);
		return op;
	}

	OpPtr from_numpy(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device)
	{
		ArrayPtr arr = Array::from_ptr(ptr, nbytes, shape, dtype, device);
		OpPtr op = std::make_shared<NumpyOp>(arr);
		return op;
	}

	OpPtr broadcast(OpPtr op, const ShapeView &view)
	{
		ArrayPtr in_arr = op->get_output();
		const Shape &in_shape = in_arr->get_shape();
		const ShapeView &in_view = in_shape.get_view();
		if (in_view == view)
		{
			return op;
		}

		std::pair<Shape, ShapeDims> broadcast_result = in_shape.broadcast(view);
		const Shape &broadcast_shape = broadcast_result.first;
		if (in_shape == broadcast_shape)
		{
			return op;
		}

		const ShapeDims &broadcast_dims = broadcast_result.second;
		ArrayPtr out_arr = Array::empty(broadcast_shape, in_arr->get_dtype(), in_arr->get_device());
		OpPtr out_op = std::make_shared<BroadcastOp>(out_arr, op, in_view, view, broadcast_dims);
		return out_op;
	}

	OpPtr broadcast_to(OpPtr op, const ShapeView &view)
	{
		ArrayPtr in_arr = op->get_output();
		const Shape &in_shape = in_arr->get_shape();
		const ShapeView &in_view = in_shape.get_view();
		if (in_view == view)
		{
			return op;
		}

		std::pair<Shape, ShapeDims> broadcast_result = in_shape.broadcast_to(view);
		const Shape &broadcast_shape = broadcast_result.first;
		if (in_shape == broadcast_shape)
		{
			return op;
		}

		const ShapeDims &broadcast_dims = broadcast_result.second;
		ArrayPtr out_arr = Array::empty(broadcast_shape, in_arr->get_dtype(), in_arr->get_device());
		OpPtr out_op = std::make_shared<BroadcastOp>(out_arr, op, in_view, view, broadcast_dims);
		return out_op;
	}

	OpPtr add(OpPtr lop, OpPtr rop)
	{
		return binary_ss<AddOp>(lop, rop);
	}

	OpPtr sub(OpPtr lop, OpPtr rop)
	{
		return binary_ss<SubOp>(lop, rop);
	}

	OpPtr mul(OpPtr lop, OpPtr rop)
	{
		return binary_ss<MulOp>(lop, rop);
	}

	OpPtr div(OpPtr lop, OpPtr rop)
	{
		return binary_ss<DivOp>(lop, rop);
	}

	OpPtr self_add(OpPtr lop, OpPtr rop)
	{
		return self_binary_ss<AddOp>(lop, rop);
	}

	OpPtr self_sub(OpPtr lop, OpPtr rop)
	{
		return self_binary_ss<SubOp>(lop, rop);
	}

	OpPtr self_mul(OpPtr lop, OpPtr rop)
	{
		return self_binary_ss<MulOp>(lop, rop);
	}

	OpPtr self_div(OpPtr lop, OpPtr rop)
	{
		return self_binary_ss<DivOp>(lop, rop);
	}
}