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

    void Op::update_grad(std::shared_ptr<Op> grad, bool sub) {
        this->grad = sub ? self_sub(this->grad, grad) : self_add(this->grad, grad);
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
        // z = x + y
        // dx += dz
        // dy -= dz
        lhs->init_grad();
        lhs->update_grad(grad);
        rhs->init_grad();
        rhs->update_grad(grad, true);
    }

    void MulOp::backward() const {
        // z = x*y
        // dx += dz*y
        // dy += dz*x
        // Use detach to prevent circular dependencies
        lhs->init_grad();
        lhs->update_grad(mul(grad, detach(rhs)));
        rhs->init_grad();
        rhs->update_grad(mul(grad, detach(lhs)));
    }

    void DivOp::backward() const {
        // z = x/y
        // dx += dz * (1/y)
        // dy += dz * (-x / y**2)
        // dy -= dz * (z / y)
        // Use detach to prevent circular dependencies
        lhs->init_grad();
        lhs->update_grad(div(grad, detach(rhs)));
        rhs->init_grad();
        OpPtr self = detach(std::const_pointer_cast<Op>(shared_from_this()));
        rhs->update_grad(mul(grad, div(self, detach(rhs))), true);
    }

    void MatmulOp::backward() const {
        // Transpose the last two dimensions of lhs and rhs
        // z = x @ y
        // dx += dz @ y^T
        // dy += x^T @ dz
        isize ndim = lhs->get_lazy()->get_ndim();
        lhs->init_grad();
        lhs->update_grad(matmul(grad, transpose(rhs, ndim - 2, ndim - 1)));
        rhs->init_grad();
        rhs->update_grad(matmul(transpose(lhs, ndim - 2, ndim - 1), grad));
    }

    void SqOp::backward() const {
        // z = x**2
        // dx += dz * 2x
        operand->init_grad();
        operand->update_grad(mul(grad, mul(detach(operand), 2.0f)));
    }

    void SqrtOp::backward() const {
        // z = sqrt(x)
        // dx += dz / (2*sqrt(x))
        // dx += dz / 2z
        operand->init_grad();
        OpPtr self = detach(std::const_pointer_cast<Op>(shared_from_this()));
        operand->update_grad(div(grad, mul(self, 2.0f)));
    }

    void NegOp::backward() const {
        // z = -x
        // dx += dz * -1
        // dx -= dz
        operand->init_grad();
        operand->update_grad(grad, true);
    }

    void IdentityOp::backward() const {
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
        OpPtr self = detach(std::const_pointer_cast<Op>(shared_from_this()));
        operand->update_grad(mul(grad, self));
    }

    void LogOp::backward() const {
        // z = log(x)
        // dx += dz / x
        operand->init_grad();
        operand->update_grad(div(grad, detach(operand)));
    }

    void RecipOp::backward() const {
        // z = 1/x
        // dx += dz * -1/x**2
        // dx += dz * -z**2
        // dx -= dz * z**2
        operand->init_grad();
        OpPtr self = detach(std::const_pointer_cast<Op>(shared_from_this()));
        operand->update_grad(mul(grad, sq(self)), true);
    }

    void SliceOp::backward() const {
        operand->init_grad();
        operand->grad_root = slice(operand->grad, ranges);
        operand->grad_root = self_add(operand->grad_root, grad);
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
            operand->update_grad(reshape(identity(grad), operand_view));
        }
    }

    void PermuteOp::backward() const {
        operand->init_grad();
        // Copy must be done before permuting to ensure gradient independence
        // Permuting array does not invoke copying
        OpPtr grad_copy = identity(grad);
        ShapeView reverse_dims = grad_copy->get_lazy()->get_shape().undo_permute_view(dims);
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
        // No need to use detach here since both operand's array and "this" array are not modified
        // Column reduction: operand's array is of shape (d1, d2) and "this" array is of shape (d1, 1)
        // All reduction: operand's array is of shape (d1, d2, etc.) and "this" array is of shape (1)
        // eq() handles broadcasting automatically
        OpPtr mask = eq(operand, std::const_pointer_cast<Op>(shared_from_this()));
        operand->update_grad(mul(astype(mask, operand->get_lazy()->get_dtype()), grad));
    }

    void MinOp::backward() const {
        operand->init_grad();
        // No need to use detach here since both operand's array and "this" array are not modified
        // Column reduction: operand's array is of shape (d1, d2) and "this" array is of shape (d1, 1)
        // All reduction: operand's array is of shape (d1, d2, etc.) and "this" array is of shape (1)
        // eq() handles broadcasting automatically
        OpPtr mask = eq(operand, std::const_pointer_cast<Op>(shared_from_this()));
        operand->update_grad(mul(astype(mask, operand->get_lazy()->get_dtype()), grad));
    }

    const std::string UnaryOp::str() const {
        return Op::str() + ", in-place: " + std::to_string(in_place) + ", operand: " + operand->get_lazy()->get_id().str();
    }

    const std::string BinaryOp::str() const {
        return Op::str() + ", in-place: " + std::to_string(in_place) + ", lhs: " + lhs->get_lazy()->get_id().str() + ", rhs: " + rhs->get_lazy()->get_id().str();
    }

    const std::string MatmulOp::str() const {
        return Op::str() + ", lhs: " + lhs->get_lazy()->get_id().str() + ", rhs: " + rhs->get_lazy()->get_id().str();
    }

    const std::string TransformOp::str() const {
        return Op::str() + ", operand: " + operand->get_lazy()->get_id().str();
    }

    const std::string ReduceOp::str() const {
        return Op::str() + ", operand: " + operand->get_lazy()->get_id().str() + ", dims: " + vnumstr(dims) + ", default value: " + std::to_string(default_val);
    }

    OpPtr detach(OpPtr op) {
        // Shared array -> shared buffer -> buffer goes out of scope -> Memory is freed twice
        // Solution: separate buffers using same memory region
        LazyArrayPtr in_arr = op->get_lazy();
        LazyArrayPtr out_arr = LazyArray::from_ptr(in_arr->get_ptr(), in_arr->get_nbytes(), in_arr->get_shape(), in_arr->get_dtype(), in_arr->get_device());
        return std::make_shared<NoopOp>(out_arr);
    }

    isize item(OpPtr op) {
        LazyArrayPtr arr = op->get_lazy();

        if (arr->get_numel() != 1) {
            throw std::runtime_error("Array " + arr->get_id().str() +
                                     " must have exactly one element but has " +
                                     std::to_string(arr->get_numel()) + " elements.");
        }

        auto iter = std::make_unique<ArrayIter>(arr);
        iter->start();
        uint8_t *ptr = iter->next();
        return arr->get_dtype()->get_low_level_value(ptr);
    }

    OpPtr full_impl(const ShapeView &view, isize c, DtypePtr dtype, DevicePtr device) {
        LazyArrayPtr arr = LazyArray::empty(Shape(view), dtype, device);
        OpPtr op = std::make_shared<FullOp>(arr, view, c, dtype);
        return op;
    }

    OpPtr zeros(const ShapeView &view, DtypePtr dtype, DevicePtr device) {
        return full(view, 0, dtype, device);
    }

    OpPtr zeros_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) {
        return full_like(in_op, 0, dtype, device);
    }

    OpPtr ones(const ShapeView &view, DtypePtr dtype, DevicePtr device) {
        return full(view, 1, dtype, device);
    }

    OpPtr ones_like(OpPtr in_op, DtypePtr dtype, DevicePtr device) {
        return full_like(in_op, 1, dtype, device);
    }

    OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, DevicePtr device) {
        LazyArrayPtr arr = LazyArray::empty(Shape(view), dtype, device);
        OpPtr op = std::make_shared<ArangeOp>(arr, view, start, step, dtype);
        return op;
    }

    OpPtr from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device) {
        LazyArrayPtr arr = LazyArray::from_ptr(ptr, nbytes, shape, dtype, device);
        OpPtr op = std::make_shared<BuffOp>(arr);
        return op;
    }

    OpPtr broadcast(OpPtr op, const ShapeView &view) {
        LazyArrayPtr in_arr = op->get_lazy();
        const Shape &in_shape = in_arr->get_shape();
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
        LazyArrayPtr out_arr = LazyArray::empty(broadcast_shape, in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<BroadcastOp>(out_arr, op, in_view, view, broadcast_dims);
        return out_op;
    }

    OpPtr broadcast_to(OpPtr op, const ShapeView &view) {
        LazyArrayPtr in_arr = op->get_lazy();
        const Shape &in_shape = in_arr->get_shape();
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
        LazyArrayPtr out_arr = LazyArray::empty(broadcast_shape, in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<BroadcastOp>(out_arr, op, in_view, view, broadcast_dims);
        return out_op;
    }

    OpPtr slice(OpPtr in_op, const RangeVec &ranges) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        LazyArrayPtr out_arr = LazyArray::empty(in_arr->get_shape().slice(ranges), in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<SliceOp>(out_arr, in_op, ranges);
        return out_op;
    }

    OpPtr astype(OpPtr in_op, DtypePtr dtype) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        if (in_arr->get_dtype() == dtype) {
            return in_op;
        }
        LazyArrayPtr out_arr = LazyArray::empty(in_arr->get_shape(), dtype, in_arr->get_device());
        OpPtr out_op = std::make_shared<AstypeOp>(out_arr, in_op, dtype);
        return out_op;
    }

    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        LazyArrayPtr out_arr = LazyArray::empty(in_arr->get_shape().unsqueeze(dims), in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<UnsqueezeOp>(out_arr, in_op, dims);
        return out_op;
    }

    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        LazyArrayPtr out_arr = LazyArray::empty(in_arr->get_shape().squeeze(dims), in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<SqueezeOp>(out_arr, in_op, dims);
        return out_op;
    }

    OpPtr add(OpPtr lop, OpPtr rop) { return binary<AddOp>(lop, rop); }

    OpPtr sub(OpPtr lop, OpPtr rop) { return binary<SubOp>(lop, rop); }

    OpPtr mul(OpPtr lop, OpPtr rop) { return binary<MulOp>(lop, rop); }

    OpPtr div(OpPtr lop, OpPtr rop) { return binary<DivOp>(lop, rop); }

    OpPtr matmul(OpPtr lop, OpPtr rop) {
        MatmulOp dummy_op(nullptr, nullptr, nullptr);
        LazyArrayPtr larr = lop->get_lazy();
        LazyArrayPtr rarr = rop->get_lazy();
        const Shape &lshape = larr->get_shape();
        const ShapeView &lview = larr->get_view();
        const ShapeView &rview = rarr->get_view();
        DtypePtr ldtype = larr->get_dtype();
        DtypePtr rdtype = rarr->get_dtype();
        DevicePtr ldevice = larr->get_device();
        DevicePtr rdevice = rarr->get_device();

        if (!lshape.matmul_broadcastable(rview)) {
            throw IncompatShapesForOp(dummy_op.get_opcode_str(), vnumstr(lview), vnumstr(rview));
        }
        if (!binary_dtypes.contains(ldtype) || ldtype != rdtype) {
            throw IncompatDtypesForOp(dummy_op.get_opcode_str(), ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice) {
            throw IncompatDevicesForOp(dummy_op.get_opcode_str(), ldevice->str(), rdevice->str());
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
        LazyArrayPtr mm_arr = LazyArray::empty(Shape(mm_view), ldtype, ldevice);
        OpPtr out_op = std::make_shared<MatmulOp>(mm_arr, mm_lop, mm_rop);

        // Reshape to expected result's shape
        ShapeView reshaped_mm_view = broadcasted_lview;
        reshaped_mm_view[reshaped_mm_view.size() - 1] = rview[rview.size() - 1];
        out_op = reshape(out_op, reshaped_mm_view);
        return out_op;
    }

    OpPtr self_add(OpPtr lop, OpPtr rop) { return self_binary<AddOp>(lop, rop); }

    OpPtr self_sub(OpPtr lop, OpPtr rop) { return self_binary<SubOp>(lop, rop); }

    OpPtr self_mul(OpPtr lop, OpPtr rop) { return self_binary<MulOp>(lop, rop); }

    OpPtr self_div(OpPtr lop, OpPtr rop) { return self_binary<DivOp>(lop, rop); }

    OpPtr eq(OpPtr lop, OpPtr rop) { return cmp<EqOp>(lop, rop, all_dtypes); }

    OpPtr neq(OpPtr lop, OpPtr rop) { return cmp<NeqOp>(lop, rop, all_dtypes); }

    OpPtr lt(OpPtr lop, OpPtr rop) { return cmp<LtOp>(lop, rop, numeric_dtypes); }

    OpPtr gt(OpPtr lop, OpPtr rop) { return cmp<GtOp>(lop, rop, numeric_dtypes); }

    OpPtr leq(OpPtr lop, OpPtr rop) { return cmp<LeqOp>(lop, rop, numeric_dtypes); }

    OpPtr geq(OpPtr lop, OpPtr rop) { return cmp<GeqOp>(lop, rop, numeric_dtypes); }

    OpPtr sq(OpPtr in_op, bool in_place) { return unary<SqOp>(in_op, in_place); }

    OpPtr sqrt(OpPtr in_op, bool in_place) { return unary_float<SqrtOp>(in_op, in_place); }

    OpPtr neg(OpPtr in_op, bool in_place) { return unary<NegOp>(in_op, in_place); }

    OpPtr identity(OpPtr in_op, bool in_place) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        LazyArrayPtr out_arr = LazyArray::empty(Shape(in_arr->get_view()), in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<IdentityOp>(out_arr, in_op);
        return out_op;
    }

    OpPtr exp(OpPtr in_op, bool in_place) { return unary_float<ExpOp>(in_op, in_place); }

    OpPtr log(OpPtr in_op, bool in_place) { return unary_float<LogOp>(in_op, in_place); }

    OpPtr recip(OpPtr in_op, bool in_place) { return unary_float<RecipOp>(in_op, in_place); }

    OpPtr reshape(OpPtr in_op, const ShapeView &view) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        if (in_arr->get_view() == view) {
            return in_op;
        }
        LazyArrayPtr out_arr = LazyArray::empty(in_arr->get_shape().reshape(view), in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<ReshapeOp>(out_arr, in_op, view);
        return out_op;
    }

    OpPtr permute(OpPtr in_op, const ShapeDims &dims) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        LazyArrayPtr out_arr = LazyArray::empty(in_arr->get_shape().permute(dims), in_arr->get_dtype(), in_arr->get_device());
        OpPtr out_op = std::make_shared<PermuteOp>(out_arr, in_op, dims);
        return out_op;
    }

    OpPtr transpose(OpPtr in_op, isize start_dim, isize end_dim) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        const Shape &in_shape = in_arr->get_shape();
        ShapeDims transpose_dims = in_shape.transpose(start_dim, end_dim);
        return permute(in_op, transpose_dims);
    }

    OpPtr flatten(OpPtr in_op, isize start_dim, isize end_dim) {
        LazyArrayPtr in_arr = in_op->get_lazy();
        const Shape &in_shape = in_arr->get_shape();
        ShapeView flattened_view = in_shape.flatten(start_dim, end_dim);
        return reshape(in_op, flattened_view);
    }

    OpPtr sum(OpPtr in_op, const ShapeDims &dims) {
        return reduce<SumOp>(in_op, dims, in_op->get_lazy()->get_dtype(), numeric_dtypes);
    }

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

    OpPtr max(OpPtr in_op, const ShapeDims &dims) {
        return reduce<MaxOp>(in_op, dims, in_op->get_lazy()->get_dtype(), numeric_dtypes);
    }

    OpPtr min(OpPtr in_op, const ShapeDims &dims) {
        return reduce<MinOp>(in_op, dims, in_op->get_lazy()->get_dtype(), numeric_dtypes);
    }

    OpPtr argmax(OpPtr in_op, const ShapeDims &dims) {
        return reduce<ArgmaxOp>(in_op, dims, &i32, numeric_dtypes);
    }

    OpPtr argmin(OpPtr in_op, const ShapeDims &dims) {
        return reduce<ArgminOp>(in_op, dims, &i32, numeric_dtypes);
    }
} // namespace ax::graph