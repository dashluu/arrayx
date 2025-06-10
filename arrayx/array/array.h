#pragma once

#include "../graph/compute_graph.h"
#include "backend.h"

namespace ax::array {
    using namespace ax::core;
    using namespace ax::runtime;
    using namespace ax::graph;

    class Array {
    private:
        OpPtr op = nullptr;
        std::shared_ptr<ComputeGraph> compute_graph = nullptr;
        bool initial_run = false;

        static DevicePtr get_backend_device(const std::string &device_name) { return Backend::instance().get_device(device_name); }
        RunnerPtr get_backend_runner() const { return Backend::instance().get_runner(op->get_lazy()->get_device_name()); }
        std::function<std::shared_ptr<ComputeGraph>(OpPtr)> get_backend_graph_builder() { return Backend::instance().get_graph_builder(op->get_lazy()->get_device_name()); }

    public:
        Array() = default;
        Array(OpPtr op) : op(op) {}

        Array(const Array &arr) {
            op = arr.op;
            compute_graph = arr.compute_graph;
            initial_run = arr.initial_run;
        }

        Array &operator=(const Array &arr) {
            op = arr.op;
            compute_graph = arr.compute_graph;
            initial_run = arr.initial_run;
            return *this;
        }

        const Id &get_id() const { return op->get_lazy()->get_id(); }
        const Shape &get_shape() const { return op->get_lazy()->get_shape(); }
        isize get_offset() const { return op->get_lazy()->get_offset(); }
        const ShapeView &get_view() const { return op->get_lazy()->get_view(); }
        const ShapeStride &get_stride() const { return op->get_lazy()->get_stride(); }
        uint8_t *get_ptr() const { return op->get_lazy()->get_ptr(); }
        DtypePtr get_dtype() const { return op->get_lazy()->get_dtype(); }
        DevicePtr get_device() const { return op->get_lazy()->get_device(); }
        std::optional<Array> get_grad() const { return op->grad == nullptr ? std::nullopt : std::optional<Array>(ax::graph::detach(op->grad)); }
        isize get_numel() const { return op->get_lazy()->get_numel(); }
        isize get_ndim() const { return op->get_lazy()->get_ndim(); }
        isize get_itemsize() const { return op->get_lazy()->get_itemsize(); }
        isize get_nbytes() const { return op->get_lazy()->get_nbytes(); }
        bool is_grad_enabled() const { return op->is_grad_enabled(); }
        void enable_grad(bool enabled = true) { op->enable_grad(enabled); }
        bool is_contiguous() const { return op->get_lazy()->is_contiguous(); }

        const std::string str() {
            eval();
            return op->get_lazy()->str();
        }

        const std::string graph_str() {
            eval();
            return compute_graph->str();
        }

        std::shared_ptr<ComputeGraph> get_graph() { return compute_graph; }

        isize item() {
            eval();
            return ax::graph::item(op);
        }

        Array detach() const { return Array(ax::graph::detach(op)); }
        void eval();
        void backward();
        void compile();

        // Initializer operations
        template <typename T>
        static Array full(const ShapeView &view, T c, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::full(view, c, dtype, device));
        }

        template <typename T>
        static Array full_like(const Array &other, T c, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::full_like(other.op, c, dtype, device));
        }

        static Array arange(const ShapeView &view, isize start, isize step, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::arange(view, start, step, dtype, device));
        }

        static Array zeros(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::zeros(view, dtype, device));
        }

        static Array ones(const ShapeView &view, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::ones(view, dtype, device));
        }

        static Array zeros_like(const Array &other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::zeros_like(other.op, dtype, device));
        }

        static Array ones_like(const Array &other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::ones_like(other.op, dtype, device));
        }

        static Array from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::from_ptr(ptr, nbytes, shape, dtype, device));
        }

        static Array empty_like(const Array &other, DtypePtr dtype = &f32, const std::string &device_name = default_device_name) {
            DevicePtr device = get_backend_device(device_name);
            return Array(ax::graph::empty_like(other.op, dtype, device));
        }

        static Array empty_twin(const Array &other) {
            return Array(ax::graph::empty_like(other.op, other.get_dtype(), other.get_device()));
        }

        // Element-wise operations
        Array operator+(const Array &rhs) const { return Array(ax::graph::add(op, rhs.op)); }

        template <Numeric T>
        Array operator+(T c) const { return Array(ax::graph::add(op, c)); }

        Array operator-(const Array &rhs) const { return Array(ax::graph::sub(op, rhs.op)); }

        template <Numeric T>
        Array operator-(T c) const { return Array(ax::graph::sub(op, c)); }

        Array operator*(const Array &rhs) const { return Array(ax::graph::mul(op, rhs.op)); }

        template <Numeric T>
        Array operator*(T c) const { return Array(ax::graph::mul(op, c)); }

        Array operator/(const Array &rhs) const { return Array(ax::graph::div(op, rhs.op)); }

        template <Numeric T>
        Array operator/(T c) const { return Array(ax::graph::div(op, c)); }

        Array operator+=(const Array &rhs) const { return Array(ax::graph::inplace_add(op, rhs.op)); }

        template <Numeric T>
        Array operator+=(T c) const { return Array(ax::graph::inplace_add(op, c)); }

        Array operator-=(const Array &rhs) const { return Array(ax::graph::inplace_sub(op, rhs.op)); }

        template <Numeric T>
        Array operator-=(T c) const { return Array(ax::graph::inplace_sub(op, c)); }

        Array operator*=(const Array &rhs) const { return Array(ax::graph::inplace_mul(op, rhs.op)); }

        template <Numeric T>
        Array operator*=(T c) const { return Array(ax::graph::inplace_mul(op, c)); }

        Array operator/=(const Array &rhs) const { return Array(ax::graph::inplace_div(op, rhs.op)); }

        template <Numeric T>
        Array operator/=(T c) const { return Array(ax::graph::inplace_div(op, c)); }

        Array matmul(const Array &rhs) const { return Array(ax::graph::matmul(op, rhs.op)); }
        Array exp(bool in_place = false) const { return Array(ax::graph::exp(op, in_place)); }
        Array log(bool in_place = false) const { return Array(ax::graph::log(op, in_place)); }
        Array sqrt(bool in_place = false) const { return Array(ax::graph::sqrt(op, in_place)); }
        Array sq(bool in_place = false) const { return Array(ax::graph::sq(op, in_place)); }
        Array neg(bool in_place = false) const { return Array(ax::graph::neg(op, in_place)); }
        Array operator-() const { return Array(ax::graph::neg(op)); }
        Array recip(bool in_place = false) const { return Array(ax::graph::recip(op, in_place)); }
        Array operator==(const Array &rhs) const { return Array(ax::graph::eq(op, rhs.op)); }
        Array operator!=(const Array &rhs) const { return Array(ax::graph::neq(op, rhs.op)); }
        Array operator<(const Array &rhs) const { return Array(ax::graph::lt(op, rhs.op)); }
        Array operator>(const Array &rhs) const { return Array(ax::graph::gt(op, rhs.op)); }
        Array operator<=(const Array &rhs) const { return Array(ax::graph::leq(op, rhs.op)); }
        Array operator>=(const Array &rhs) const { return Array(ax::graph::geq(op, rhs.op)); }
        Array minimum(const Array &rhs) const { return Array(ax::graph::minimum(op, rhs.op)); }
        Array maximum(const Array &rhs) const { return Array(ax::graph::maximum(op, rhs.op)); }

        template <Numeric T>
        Array operator==(T c) const { return Array(ax::graph::eq(op, c)); }

        template <Numeric T>
        Array operator!=(T c) const { return Array(ax::graph::neq(op, c)); }

        template <Numeric T>
        Array operator<(T c) const { return Array(ax::graph::lt(op, c)); }

        template <Numeric T>
        Array operator>(T c) const { return Array(ax::graph::gt(op, c)); }

        template <Numeric T>
        Array operator<=(T c) const { return Array(ax::graph::leq(op, c)); }

        template <Numeric T>
        Array operator>=(T c) const { return Array(ax::graph::geq(op, c)); }

        template <Numeric T>
        Array minimum(T c) const { return Array(ax::graph::minimum(op, c)); }

        template <Numeric T>
        Array maximum(T c) const { return Array(ax::graph::maximum(op, c)); }

        // Reduction operations
        Array sum(const ShapeDims &dims = {}) const { return Array(ax::graph::sum(op, dims)); }
        Array mean(const ShapeDims &dims = {}) const { return Array(ax::graph::mean(op, dims)); }
        Array max(const ShapeDims &dims = {}) const { return Array(ax::graph::max(op, dims)); }
        Array min(const ShapeDims &dims = {}) const { return Array(ax::graph::min(op, dims)); }
        Array argmax(const ShapeDims &dims = {}) const { return Array(ax::graph::argmax(op, dims)); }
        Array argmin(const ShapeDims &dims = {}) const { return Array(ax::graph::argmin(op, dims)); }

        // Shape operations
        Array broadcast(const ShapeView &view) const { return Array(ax::graph::broadcast(op, view)); }
        Array broadcast_to(const ShapeView &view) const { return Array(ax::graph::broadcast_to(op, view)); }
        Array slice(const RangeVec &ranges) const { return Array(ax::graph::slice(op, ranges)); }
        Array reshape(const ShapeView &view) const { return Array(ax::graph::reshape(op, view)); }
        Array flatten(isize start_dim, isize end_dim) const { return Array(ax::graph::flatten(op, start_dim, end_dim)); }
        Array squeeze(const ShapeDims &dims = {}) const { return Array(ax::graph::squeeze(op, dims)); }
        Array unsqueeze(const ShapeDims &dims = {}) const { return Array(ax::graph::unsqueeze(op, dims)); }
        Array permute(const ShapeDims &dims) const { return Array(ax::graph::permute(op, dims)); }
        Array transpose(isize start_dim, isize end_dim) const { return Array(ax::graph::transpose(op, start_dim, end_dim)); }

        // Type operations
        Array astype(DtypePtr dtype) const { return Array(ax::graph::astype(op, dtype)); }
    };

    template <Numeric T>
    Array operator+(T c, const Array &arr) {
        return arr + c;
    }

    template <Numeric T>
    Array operator-(T c, const Array &arr) {
        return arr - c;
    }

    template <Numeric T>
    Array operator*(T c, const Array &arr) {
        return arr * c;
    }

    template <Numeric T>
    Array operator/(T c, const Array &arr) {
        return arr.recip() * c;
    }

    using ArrayVec = std::vector<Array>;
} // namespace ax::array