#include "ops.h"

namespace ax::graph {
    void Op::init_grad(bool with_zeros) {
        if (grad == nullptr) {
            DtypePtr dtype = lazy->get_dtype();
            if (dtype->get_type() != DtypeType::FLOAT) {
                throw std::runtime_error("Only arrays of floating-point types can have gradients but array " + lazy->get_id().str() + " has type " + dtype->str());
            }
            const ShapeView &view = lazy->get_shape().get_view();
            DtypePtr grad_dtype = float_dtype_by_dtype.at(dtype);
            DevicePtr device = lazy->get_device();
            grad = with_zeros ? zeros(view, grad_dtype, device) : ones(view, grad_dtype, device);
            grad_root = grad;
        }
    }

    void Op::update_grad(OpPtr grad, bool sub) {
        this->grad = sub ? inplace_sub(this->grad, grad) : inplace_add(this->grad, grad);
        this->grad_root = this->grad;
    }

    void AddOp::backward() const {
        // In-place or not, gradient should be computed properly
        // z = x + y
        // dx += dz
        // dy += dz
        lhs->init_grad();
        lhs->update_grad(grad);
        rhs->init_grad();
        rhs->update_grad(grad);
    }

    void SubOp::backward() const {
        // z = x - y
        // dx += dz
        // dy -= dz
        lhs->init_grad();
        lhs->update_grad(grad);
        rhs->init_grad();
        rhs->update_grad(grad, true);
    }

    void MulOp::backward() const {
        // z = x * y
        // dx += dz * y
        // dy += dz * x
        // Use detach to prevent circular dependencies
        lhs->init_grad();
        lhs->update_grad(mul(grad, de_rhs()));
        rhs->init_grad();
        rhs->update_grad(mul(grad, de_lhs()));
    }

    void DivOp::backward() const {
        // z = x / y
        // dx += dz * (1/y)
        // dy += dz * (-x/y**2)
        // dy -= dz * (z/y)
        // Use detach to prevent circular dependencies
        lhs->init_grad();
        lhs->update_grad(div(grad, de_rhs()));
        rhs->init_grad();
        rhs->update_grad(mul(grad, div(de_op(), de_rhs())), true);
    }

    void MinimumOp::backward() const {
        // z = min(x, y)
        // dx += dz * (1 where x is min and 0 otherwise)
        // dy += dz * (1 where y is min and 0 otherwise)
        OpPtr out_op = de_op();
        OpPtr lminimum = astype(eq(de_lhs(), out_op), out_op->get_lazy()->get_dtype());
        OpPtr rminimum = astype(eq(de_rhs(), out_op), out_op->get_lazy()->get_dtype());
        lhs->init_grad();
        lhs->update_grad(mul(grad, lminimum));
        rhs->init_grad();
        rhs->update_grad(mul(grad, rminimum));
    }

    void MaximumOp::backward() const {
        // z = max(x, y)
        // dx += dz * (1 where x is max and 0 otherwise)
        // dy += dz * (1 where y is max and 0 otherwise)
        OpPtr out_op = de_op();
        OpPtr lmaximum = astype(eq(de_lhs(), out_op), out_op->get_lazy()->get_dtype());
        OpPtr rmaximum = astype(eq(de_rhs(), out_op), out_op->get_lazy()->get_dtype());
        lhs->init_grad();
        lhs->update_grad(mul(grad, lmaximum));
        rhs->init_grad();
        rhs->update_grad(mul(grad, rmaximum));
    }

    void MatmulOp::backward() const {
        // Transpose the last two dimensions of lhs and rhs
        // z = x @ y
        // dx += dz @ y^T
        // dy += x^T @ dz
        isize ndim = lhs->get_lazy()->get_ndim();
        lhs->init_grad();
        lhs->update_grad(matmul(grad, transpose(de_rhs(), ndim - 2, ndim - 1)));
        rhs->init_grad();
        rhs->update_grad(matmul(transpose(de_lhs(), ndim - 2, ndim - 1), grad));
    }

    void SqOp::backward() const {
        // z = x**2
        // dx += dz * (2*x)
        operand->init_grad();
        operand->update_grad(mul(grad, mul(de_operand(), 2.0f)));
    }

    void SqrtOp::backward() const {
        // z = sqrt(x)
        // dx += dz / (2*sqrt(x))
        // dx += dz / (2*z)
        operand->init_grad();
        operand->update_grad(div(grad, mul(de_op(), 2.0f)));
    }

    void NegOp::backward() const {
        // z = -x
        // dx += dz * -1
        // dx -= dz
        operand->init_grad();
        operand->update_grad(grad, true);
    }

    void CopyOp::backward() const {
        // z = x
        // dx += dz
        operand->init_grad();
        operand->update_grad(grad);
    }

    void ExpOp::backward() const {
        // z = exp(x)
        // dx += dz * exp(x)
        // dx += dz * z
        operand->init_grad();
        operand->update_grad(mul(grad, de_op()));
    }

    void LogOp::backward() const {
        // z = log(x)
        // dx += dz / x
        operand->init_grad();
        operand->update_grad(div(grad, de_operand()));
    }

    void RecipOp::backward() const {
        // z = 1/x
        // dx += dz * -1/x**2
        // dx += dz * -z**2
        // dx -= dz * z**2
        operand->init_grad();
        operand->update_grad(mul(grad, sq(de_op())), true);
    }

    void SliceOp::backward() const {
        operand->init_grad();
        operand->grad_root = slice(operand->grad, ranges);
        operand->grad_root = inplace_add(operand->grad_root, grad);
    }

    void ReshapeOp::backward() const {
        operand->init_grad();
        const ShapeView &operand_view = operand->get_lazy()->get_view();

        // Copy must be done to ensure gradient independence
        if (grad->get_lazy()->copy_when_reshape(operand_view)) {
            // No need to copy since reshaping invokes copying anyway
            operand->update_grad(reshape(grad, operand_view));
        } else {
            // Copy first and then reshape because reshaping to incompatible shape might cause another copy
            // Copy first creates a contiguous array so reshaping it is easier
            operand->update_grad(reshape(copy(grad), operand_view));
        }
    }

    void PermuteOp::backward() const {
        operand->init_grad();
        // Copy must be done before permuting to ensure gradient independence
        // Permuting array does not invoke copying
        OpPtr grad_copy = copy(grad);
        const ShapeView &reverse_dims = grad_copy->get_lazy()->get_shape().undo_permute_view(dims);
        operand->update_grad(permute(grad_copy, reverse_dims));
    }

    void BroadcastOp::backward() const {
        operand->init_grad();
        operand->update_grad(reshape(sum(grad, dims), input_view));
    }

    void SqueezeOp::backward() const {
        operand->init_grad();
        operand->update_grad(unsqueeze(grad, dims));
    }

    void UnsqueezeOp::backward() const {
        operand->init_grad();
        operand->update_grad(squeeze(grad, dims));
    }

    void SumOp::backward() const {
        operand->init_grad();
        operand->update_grad(grad);
    }

    void MaxOp::backward() const {
        operand->init_grad();
        // Column reduction: operand's array is of shape (d1, d2) and "this" array is of shape (d1, 1)
        // All reduction: operand's array is of shape (d1, d2, etc.) and "this" array is of shape (1)
        // eq() handles broadcasting automatically
        OpPtr mask = eq(operand, de_op());
        operand->update_grad(mul(astype(mask, operand->get_lazy()->get_dtype()), grad));
    }

    void MinOp::backward() const {
        operand->init_grad();
        // Column reduction: operand's array is of shape (d1, d2) and "this" array is of shape (d1, 1)
        // All reduction: operand's array is of shape (d1, d2, etc.) and "this" array is of shape (1)
        // eq() handles broadcasting automatically
        OpPtr mask = eq(operand, de_op());
        operand->update_grad(mul(astype(mask, operand->get_lazy()->get_dtype()), grad));
    }

    OpPtr detach(OpPtr op) {
        // Shared array -> shared buffer -> buffer goes out of scope -> Memory is freed twice
        // Solution: separate buffers using same memory region
        LazyPtr in_lazy = op->get_lazy();
        LazyPtr out_lazy = Lazy::from_ptr(in_lazy->get_ptr(), in_lazy->get_nbytes(), in_lazy->get_shape(), in_lazy->get_dtype(), in_lazy->get_device());
        return std::make_shared<Nop>(out_lazy);
    }

    OpPtr empty_like(OpPtr op, DtypePtr dtype, DevicePtr device) {
        LazyPtr in_lazy = op->get_lazy();
        LazyPtr out_lazy = Lazy::empty(in_lazy->get_shape(), dtype, device);
        return std::make_shared<Nop>(out_lazy);
    }

    isize item(OpPtr op) {
        LazyPtr lazy = op->get_lazy();
        if (lazy->get_numel() != 1) {
            throw std::runtime_error("Array " + lazy->get_id().str() + " must have exactly one element but has " + std::to_string(lazy->get_numel()) + " elements.");
        }
        auto iter = std::make_unique<LazyIter>(lazy);
        iter->start();
        uint8_t *ptr = iter->next();
        return lazy->get_dtype()->get_low_level_value(ptr);
    }

    OpPtr zeros(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return full(view, 0, dtype, device); }
    OpPtr zeros_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) { return full_like(in_op, 0, dtype, device); }
    OpPtr ones(const ShapeView &view, DtypePtr dtype, DevicePtr device) { return full(view, 1, dtype, device); }
    OpPtr ones_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) { return full_like(in_op, 1, dtype, device); }

    OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, DevicePtr device) {
        LazyPtr lazy = Lazy::empty(Shape(view), dtype, device);
        OpPtr op = std::make_shared<ArangeOp>(lazy, view, start, step, dtype);
        return op;
    }

    OpPtr from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device) {
        LazyPtr lazy = Lazy::from_ptr(ptr, nbytes, shape, dtype, device);
        OpPtr op = std::make_shared<Nop>(lazy);
        return op;
    }

    OpPtr broadcast(OpPtr op, const ShapeView &view) {
        LazyPtr in_lazy = op->get_lazy();
        const Shape &in_shape = in_lazy->get_shape();
        const ShapeView &in_view = in_shape.get_view();
        if (in_view == view) {
            return op;
        }

        std::pair<Shape, ShapeDims> broadcast_result = in_shape.broadcast(view);
        const Shape &broadcast_shape = broadcast_result.first;
        if (in_shape == broadcast_shape) {
            return op;
        }

        const ShapeDims &broadcast_dims = broadcast_result.second;
        LazyPtr out_lazy = Lazy::empty(broadcast_shape, in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<BroadcastOp>(out_lazy, op, in_view, view, broadcast_dims);
        return out_op;
    }

    OpPtr broadcast_to(OpPtr op, const ShapeView &view) {
        LazyPtr in_lazy = op->get_lazy();
        const Shape &in_shape = in_lazy->get_shape();
        const ShapeView &in_view = in_shape.get_view();
        if (in_view == view) {
            return op;
        }

        std::pair<Shape, ShapeDims> broadcast_result = in_shape.broadcast_to(view);
        const Shape &broadcast_shape = broadcast_result.first;
        if (in_shape == broadcast_shape) {
            return op;
        }

        const ShapeDims &broadcast_dims = broadcast_result.second;
        LazyPtr out_lazy = Lazy::empty(broadcast_shape, in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<BroadcastOp>(out_lazy, op, in_view, view, broadcast_dims);
        return out_op;
    }

    OpPtr slice(OpPtr in_op, const RangeVec &ranges) {
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = Lazy::empty(in_lazy->get_shape().slice(ranges), in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<SliceOp>(out_lazy, in_op, ranges);
        return out_op;
    }

    OpPtr astype(OpPtr in_op, DtypePtr dtype) {
        LazyPtr in_lazy = in_op->get_lazy();
        if (in_lazy->get_dtype() == dtype) {
            return in_op;
        }
        LazyPtr out_lazy = Lazy::empty(in_lazy->get_shape(), dtype, in_lazy->get_device());
        OpPtr out_op = std::make_shared<AstypeOp>(out_lazy, in_op, dtype);
        return out_op;
    }

    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims) {
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = Lazy::empty(in_lazy->get_shape().unsqueeze(dims), in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<UnsqueezeOp>(out_lazy, in_op, dims);
        return out_op;
    }

    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims) {
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = Lazy::empty(in_lazy->get_shape().squeeze(dims), in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<SqueezeOp>(out_lazy, in_op, dims);
        return out_op;
    }

    OpPtr add(OpPtr lop, OpPtr rop) { return elmwise_binary<AddOp>(lop, rop); }
    OpPtr sub(OpPtr lop, OpPtr rop) { return elmwise_binary<SubOp>(lop, rop); }
    OpPtr mul(OpPtr lop, OpPtr rop) { return elmwise_binary<MulOp>(lop, rop); }
    OpPtr div(OpPtr lop, OpPtr rop) { return elmwise_binary<DivOp>(lop, rop); }

    OpPtr matmul(OpPtr lop, OpPtr rop) {
        LazyPtr llazy = lop->get_lazy();
        LazyPtr rlazy = rop->get_lazy();
        const Shape &lshape = llazy->get_shape();
        const ShapeView &lview = llazy->get_view();
        const ShapeView &rview = rlazy->get_view();
        DtypePtr ldtype = llazy->get_dtype();
        DtypePtr rdtype = rlazy->get_dtype();
        DevicePtr ldevice = llazy->get_device();
        DevicePtr rdevice = rlazy->get_device();

        if (!lshape.matmul_broadcastable(rview)) {
            throw IncompatShapesForOp(MatmulOp::opname, vnumstr(lview), vnumstr(rview));
        }
        if (!binary_dtypes.contains(ldtype) || ldtype != rdtype) {
            throw IncompatDtypesForOp(MatmulOp::opname, ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice) {
            throw IncompatDevicesForOp(MatmulOp::opname, ldevice->str(), rdevice->str());
        }

        ShapeView broadcasted_lview = lview;
        ShapeView broadcasted_rview = rview;
        size_t ndim = std::max(broadcasted_lview.size(), broadcasted_rview.size());
        broadcasted_lview.insert(broadcasted_lview.begin(), ndim - broadcasted_lview.size(), 1);
        broadcasted_rview.insert(broadcasted_rview.begin(), ndim - broadcasted_rview.size(), 1);

        for (size_t i = 0; i < ndim - 2; i++) {
            isize shared_dim = std::max(broadcasted_lview[i], broadcasted_rview[i]);
            broadcasted_lview[i] = shared_dim;
            broadcasted_rview[i] = shared_dim;
        }

        isize batch = std::accumulate(
            broadcasted_lview.begin(),
            std::prev(broadcasted_lview.end(), 2),
            1,
            std::multiplies<isize>());
        ShapeView mm_lview = {
            batch,
            broadcasted_lview[broadcasted_lview.size() - 2],
            broadcasted_lview[broadcasted_lview.size() - 1]};
        ShapeView mm_rview = {
            batch,
            broadcasted_rview[broadcasted_rview.size() - 2],
            broadcasted_rview[broadcasted_rview.size() - 1]};

        // Broadcast lhs and rhs to have only 3D
        // Lhs's shape: B, M, N
        OpPtr mm_lop = broadcast(lop, broadcasted_lview);
        mm_lop = reshape(mm_lop, mm_lview);
        // Rhs's shape: B, N, K
        OpPtr mm_rop = broadcast(rop, broadcasted_rview);
        mm_rop = reshape(mm_rop, mm_rview);

        // Result's shape: B, M, K
        ShapeView mm_view = mm_lop->get_lazy()->get_view();
        mm_view[mm_view.size() - 1] = rview[rview.size() - 1];
        LazyPtr mm_arr = Lazy::empty(Shape(mm_view), ldtype, ldevice);
        OpPtr out_op = std::make_shared<MatmulOp>(mm_arr, mm_lop, mm_rop);

        // Reshape to expected result's shape
        ShapeView reshaped_mm_view = broadcasted_lview;
        reshaped_mm_view[reshaped_mm_view.size() - 1] = rview[rview.size() - 1];
        out_op = reshape(out_op, reshaped_mm_view);
        return out_op;
    }

    OpPtr inplace_add(OpPtr lop, OpPtr rop) { return inplace_binary<AddOp>(lop, rop); }
    OpPtr inplace_sub(OpPtr lop, OpPtr rop) { return inplace_binary<SubOp>(lop, rop); }
    OpPtr inplace_mul(OpPtr lop, OpPtr rop) { return inplace_binary<MulOp>(lop, rop); }
    OpPtr inplace_div(OpPtr lop, OpPtr rop) { return inplace_binary<DivOp>(lop, rop); }
    OpPtr eq(OpPtr lop, OpPtr rop) { return cmp<EqOp>(lop, rop, all_dtypes); }
    OpPtr neq(OpPtr lop, OpPtr rop) { return cmp<NeqOp>(lop, rop, all_dtypes); }
    OpPtr lt(OpPtr lop, OpPtr rop) { return cmp<LtOp>(lop, rop, numeric_dtypes); }
    OpPtr gt(OpPtr lop, OpPtr rop) { return cmp<GtOp>(lop, rop, numeric_dtypes); }
    OpPtr leq(OpPtr lop, OpPtr rop) { return cmp<LeqOp>(lop, rop, numeric_dtypes); }
    OpPtr geq(OpPtr lop, OpPtr rop) { return cmp<GeqOp>(lop, rop, numeric_dtypes); }
    OpPtr minimum(OpPtr lop, OpPtr rop) { return elmwise_binary<MinimumOp>(lop, rop); }
    OpPtr maximum(OpPtr lop, OpPtr rop) { return elmwise_binary<MaximumOp>(lop, rop); }
    OpPtr sq(OpPtr in_op, bool in_place) { return unary<SqOp>(in_op, in_place); }
    OpPtr sqrt(OpPtr in_op, bool in_place) { return unary_float<SqrtOp>(in_op, in_place); }
    OpPtr neg(OpPtr in_op, bool in_place) { return unary<NegOp>(in_op, in_place); }

    OpPtr copy(OpPtr in_op) {
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = Lazy::empty(Shape(in_lazy->get_view()), in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<CopyOp>(out_lazy, in_op);
        return out_op;
    }

    OpPtr exp(OpPtr in_op, bool in_place) { return unary_float<ExpOp>(in_op, in_place); }
    OpPtr log(OpPtr in_op, bool in_place) { return unary_float<LogOp>(in_op, in_place); }
    OpPtr recip(OpPtr in_op, bool in_place) { return unary_float<RecipOp>(in_op, in_place); }

    OpPtr reshape(OpPtr in_op, const ShapeView &view) {
        LazyPtr in_lazy = in_op->get_lazy();
        if (in_lazy->get_view() == view) {
            return in_op;
        }
        LazyPtr out_lazy = Lazy::empty(in_lazy->get_shape().reshape(view), in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<ReshapeOp>(out_lazy, in_op, view);
        return out_op;
    }

    OpPtr permute(OpPtr in_op, const ShapeDims &dims) {
        LazyPtr in_lazy = in_op->get_lazy();
        LazyPtr out_lazy = Lazy::empty(in_lazy->get_shape().permute(dims), in_lazy->get_dtype(), in_lazy->get_device());
        OpPtr out_op = std::make_shared<PermuteOp>(out_lazy, in_op, dims);
        return out_op;
    }

    OpPtr transpose(OpPtr in_op, isize start_dim, isize end_dim) {
        LazyPtr in_lazy = in_op->get_lazy();
        const ShapeDims &transpose_dims = in_lazy->get_shape().transpose(start_dim, end_dim);
        return permute(in_op, transpose_dims);
    }

    OpPtr flatten(OpPtr in_op, isize start_dim, isize end_dim) {
        LazyPtr in_lazy = in_op->get_lazy();
        const ShapeView &flattened_view = in_lazy->get_shape().flatten(start_dim, end_dim);
        return reshape(in_op, flattened_view);
    }

    OpPtr sum(OpPtr in_op, const ShapeDims &dims) { return reduce<SumOp>(in_op, dims, in_op->get_lazy()->get_dtype(), numeric_dtypes); }

    OpPtr mean(OpPtr in_op, const ShapeDims &dims) {
        OpPtr sum_op = sum(in_op, dims);
        isize numel;

        if (dims.empty()) {
            numel = in_op->get_lazy()->get_numel();
        } else {
            ShapeView view(dims.size());
            std::transform(dims.begin(), dims.end(), view.begin(), [&](isize dim) { return in_op->get_lazy()->get_shape()[dim]; });
            numel = std::accumulate(view.begin(), view.end(), 1, std::multiplies<isize>());
        }

        return div(sum_op, numel);
    }

    OpPtr max(OpPtr in_op, const ShapeDims &dims) { return reduce<MaxOp>(in_op, dims, in_op->get_lazy()->get_dtype(), numeric_dtypes); }
    OpPtr min(OpPtr in_op, const ShapeDims &dims) { return reduce<MinOp>(in_op, dims, in_op->get_lazy()->get_dtype(), numeric_dtypes); }
    OpPtr argmax(OpPtr in_op, const ShapeDims &dims) { return reduce<ArgmaxOp>(in_op, dims, &i32, numeric_dtypes); }
    OpPtr argmin(OpPtr in_op, const ShapeDims &dims) { return reduce<ArgminOp>(in_op, dims, &i32, numeric_dtypes); }
} // namespace ax::graph