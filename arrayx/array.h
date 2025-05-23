#pragma once

#include "graph/compute_graph.h"
#include "backend.h"

namespace ax
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

		const Shape &get_shape() const { return op->get_lazy()->get_shape(); }

		isize get_offset() const { return op->get_lazy()->get_offset(); }

		const ShapeView &get_view() const { return op->get_lazy()->get_view(); }

		const ShapeStride &get_stride() const { return op->get_lazy()->get_stride(); }

		uint8_t *get_ptr() const { return op->get_lazy()->get_ptr(); }

		DtypePtr get_dtype() const { return op->get_lazy()->get_dtype(); }

		DevicePtr get_device() const { return op->get_lazy()->get_device(); }

		isize get_numel() const { return op->get_lazy()->get_numel(); }

		isize get_ndim() const { return op->get_lazy()->get_ndim(); }

		isize get_itemsize() const { return op->get_lazy()->get_itemsize(); }

		isize get_nbytes() const { return op->get_lazy()->get_nbytes(); }

		bool is_contiguous() const { return op->get_lazy()->is_contiguous(); }

		static ArrayPtr full(const ShapeView &view, int c, DtypePtr dtype = &f32, const std::string &device_name = "cpu");

		static ArrayPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype = &f32, const std::string &device_name = "cpu");

		ArrayPtr add(ArrayPtr rhs) const;

		ArrayPtr sub(ArrayPtr rhs) const;

		ArrayPtr mul(ArrayPtr rhs) const;

		ArrayPtr div(ArrayPtr rhs) const;

		const std::string str() const override { return op->get_lazy()->str(); }

		void eval();

		void backward();
	};
}