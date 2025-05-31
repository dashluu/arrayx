#pragma once

#include "../graph/compute_graph.h"
#include "backend.h"

namespace ax::array
{
	using namespace ax::core;
	using namespace ax::runtime;
	using namespace ax::graph;
	class Array;
	using ArrayPtr = std::shared_ptr<Array>;

	class Array : public std::enable_shared_from_this<Array>, public IStr
	{
	private:
		OpPtr op;
		std::shared_ptr<ComputeGraph> compute_graph = nullptr;

		static DevicePtr get_backend_device(const std::string &device_name) { return Backend::instance().get_device(device_name); }

		RunnerPtr get_backend_runner() const { return Backend::instance().get_runner(op->get_lazy()->get_device_name()); }

	public:
		Array(OpPtr op) : op(op) {}

		Array(const Array &) = delete;

		Array &operator=(const Array &) = delete;

		const Id &get_id() const { return op->get_lazy()->get_id(); }

		const Shape &get_shape() const { return op->get_lazy()->get_shape(); }

		isize get_offset() const { return op->get_lazy()->get_offset(); }

		const ShapeView &get_view() const { return op->get_lazy()->get_view(); }

		const ShapeStride &get_stride() const { return op->get_lazy()->get_stride(); }

		uint8_t *get_ptr() const { return op->get_lazy()->get_ptr(); }

		DtypePtr get_dtype() const { return op->get_lazy()->get_dtype(); }

		DevicePtr get_device() const { return op->get_lazy()->get_device(); }

		ArrayPtr get_grad() const { return std::make_shared<Array>(op->grad); }

		isize get_numel() const { return op->get_lazy()->get_numel(); }

		isize get_ndim() const { return op->get_lazy()->get_ndim(); }

		isize get_itemsize() const { return op->get_lazy()->get_itemsize(); }

		isize get_nbytes() const { return op->get_lazy()->get_nbytes(); }

		bool is_grad_enabled() const { return op->grad_enabled; }

		void enable_grad(bool grad_enabled = true) { op->grad_enabled = grad_enabled; }

		bool is_contiguous() const { return op->get_lazy()->is_contiguous(); }

		const std::string str() const override { return op->get_lazy()->str(); }

		isize item() const { return ax::graph::item(op); }

		void eval();

		void backward();

		// Initializer operations
		template <typename T>
		static ArrayPtr full(const ShapeView &view, T c, DtypePtr dtype = &f32, const std::string &device_name = default_device_name)
		{
			DevicePtr device = get_backend_device(device_name);
			OpPtr out_op = ax::graph::full(view, c, dtype, device);
			return std::make_shared<Array>(out_op);
		}

		template <typename T>
		static ArrayPtr full_like(ArrayPtr other, T c, DtypePtr dtype = &f32, const std::string &device_name = default_device_name)
		{
			DevicePtr device = get_backend_device(device_name);
			OpPtr out_op = ax::graph::full_like(other->op, c, dtype, device);
			return std::make_shared<Array>(out_op);
		}

		static ArrayPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype = &f32, const std::string &device_name = default_device_name);
		static ArrayPtr zeros(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name);
		static ArrayPtr ones(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name);
		static ArrayPtr zeros_like(ArrayPtr other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name);
		static ArrayPtr ones_like(ArrayPtr other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name);
		static ArrayPtr from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype = &f32, const std::string &device_name = default_device_name);

		// Element-wise operations
		ArrayPtr add(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr add(T c) const
		{
			OpPtr out_op = ax::graph::add(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr sub(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr sub(T c) const
		{
			OpPtr out_op = ax::graph::sub(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr mul(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr mul(T c) const
		{
			OpPtr out_op = ax::graph::mul(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr div(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr div(T c) const
		{
			OpPtr out_op = ax::graph::div(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr self_add(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr self_add(T c) const
		{
			OpPtr out_op = ax::graph::self_add(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr self_sub(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr self_sub(T c) const
		{
			OpPtr out_op = ax::graph::self_sub(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr self_mul(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr self_mul(T c) const
		{
			OpPtr out_op = ax::graph::self_mul(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr self_div(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr self_div(T c) const
		{
			OpPtr out_op = ax::graph::self_div(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr matmul(ArrayPtr rhs) const;
		ArrayPtr exp(bool in_place = false) const;
		ArrayPtr log(bool in_place = false) const;
		ArrayPtr sqrt(bool in_place = false) const;
		ArrayPtr sq(bool in_place = false) const;
		ArrayPtr neg(bool in_place = false) const;
		ArrayPtr recip(bool in_place = false) const;

		// Comparison operations
		ArrayPtr eq(ArrayPtr rhs) const;

		template <NumericOrBool T>
		ArrayPtr eq(T c) const
		{
			OpPtr out_op = ax::graph::eq(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr neq(ArrayPtr rhs) const;

		template <NumericOrBool T>
		ArrayPtr neq(T c) const
		{
			OpPtr out_op = ax::graph::neq(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr lt(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr lt(T c) const
		{
			OpPtr out_op = ax::graph::lt(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr gt(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr gt(T c) const
		{
			OpPtr out_op = ax::graph::gt(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr leq(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr leq(T c) const
		{
			OpPtr out_op = ax::graph::leq(op, c);
			return std::make_shared<Array>(out_op);
		}

		ArrayPtr geq(ArrayPtr rhs) const;

		template <Numeric T>
		ArrayPtr geq(T c) const
		{
			OpPtr out_op = ax::graph::geq(op, c);
			return std::make_shared<Array>(out_op);
		}

		// Reduction operations
		ArrayPtr sum(const ShapeDims &dims = {}) const;
		ArrayPtr mean(const ShapeDims &dims = {}) const;
		ArrayPtr max(const ShapeDims &dims = {}) const;
		ArrayPtr min(const ShapeDims &dims = {}) const;
		ArrayPtr argmax(const ShapeDims &dims = {}) const;
		ArrayPtr argmin(const ShapeDims &dims = {}) const;

		// Shape operations
		ArrayPtr broadcast(const ShapeView &view) const;
		ArrayPtr broadcast_to(const ShapeView &view) const;
		ArrayPtr slice(const RangeVec &ranges) const;
		ArrayPtr reshape(const ShapeView &view) const;
		ArrayPtr flatten(isize start_dim, isize end_dim) const;
		ArrayPtr squeeze(const ShapeDims &dims = {}) const;
		ArrayPtr unsqueeze(const ShapeDims &dims = {}) const;
		ArrayPtr permute(const ShapeDims &dims) const;
		ArrayPtr transpose(isize start_dim, isize end_dim) const;

		// Type operations
		ArrayPtr astype(DtypePtr dtype) const;
	};
}