#pragma once

#include "../utils.h"
#include "../core/lazy_array.h"
#include "../device/device.h"

namespace ax::graph
{
    using namespace ax::core;
    using namespace ax::device;

    enum struct Opcode
    {
        NOOP,
        RANDN,
        ARANGE,
        FULL,
        BUFF,
        ADD,
        SUB,
        MUL,
        DIV,
        EQ,
        NEQ,
        GT,
        GEQ,
        LT,
        LEQ,
        MATMUL,
        SQ,
        SQRT,
        NEG,
        IDENTITY,
        EXP,
        LOG,
        RECIP,
        RESHAPE,
        PERMUTE,
        BROADCAST,
        SQUEEZE,
        UNSQUEEZE,
        SLICE,
        SUM,
        MAX,
        MIN,
        ARGMAX,
        ARGMIN,
        ASTYPE
    };

    enum struct Optype
    {
        INITIALIZER,
        UNARY,
        BINARY,
        MATMUL,
        TRANSFORM,
        REDUCE
    };

    enum struct ReduceMode
    {
        VALUE,
        ARG,
    };

    inline const std::unordered_map<Opcode, const std::string> str_by_opname = {
        {Opcode::NOOP, "noop"},
        {Opcode::RANDN, "randn"},
        {Opcode::ARANGE, "arange"},
        {Opcode::FULL, "full"},
        {Opcode::BUFF, "buff"},
        {Opcode::ADD, "add"},
        {Opcode::SUB, "sub"},
        {Opcode::MUL, "mul"},
        {Opcode::DIV, "div"},
        {Opcode::EQ, "eq"},
        {Opcode::NEQ, "neq"},
        {Opcode::GT, "gt"},
        {Opcode::GEQ, "geq"},
        {Opcode::LT, "lt"},
        {Opcode::LEQ, "leq"},
        {Opcode::MATMUL, "matmul"},
        {Opcode::SQ, "sq"},
        {Opcode::SQRT, "sqrt"},
        {Opcode::NEG, "neg"},
        {Opcode::IDENTITY, "identity"},
        {Opcode::EXP, "exp"},
        {Opcode::LOG, "log"},
        {Opcode::RECIP, "recip"},
        {Opcode::BROADCAST, "broadcast"},
        {Opcode::SQUEEZE, "squeeze"},
        {Opcode::UNSQUEEZE, "unsqueeze"},
        {Opcode::RESHAPE, "reshape"},
        {Opcode::PERMUTE, "permute"},
        {Opcode::SLICE, "slice"},
        {Opcode::SUM, "sum"},
        {Opcode::MAX, "max"},
        {Opcode::MIN, "min"},
        {Opcode::ARGMAX, "argmax"},
        {Opcode::ARGMIN, "argmin"}};

    struct Op : public std::enable_shared_from_this<Op>, public IStr
    {
    protected:
        Opcode opcode;
        Optype optype;
        LazyArrayPtr lazy;

    public:
        std::shared_ptr<Op> grad = nullptr;
        std::shared_ptr<Op> gradroot = nullptr;

        Op(Opcode opcode, Optype optype, LazyArrayPtr lazy) : opcode(opcode), optype(optype), lazy(lazy) {}
        Op(const Op &) = delete;
        Op &operator=(const Op &) = delete;
        virtual ~Op() = default;
        Opcode get_opcode() const { return opcode; }
        const std::string &get_opcode_str() const { return str_by_opname.at(opcode); }
        Optype get_optype() const { return optype; }
        LazyArrayPtr get_lazy() const { return lazy; }
        virtual void backward() const {}
        void init_grad(bool with_zeros = true);
        void update_grad(std::shared_ptr<Op> grad, bool sub = false);
        const std::string str() const override
        {
            return lazy->get_id().str() +
                   ": opcode: " + get_opcode_str() +
                   ", shape: " + lazy->get_shape().str();
        }
    };

    using OpPtr = std::shared_ptr<Op>;

    struct InitializerOp : public Op
    {
    public:
        InitializerOp(Opcode opcode, LazyArrayPtr lazy) : Op(opcode, Optype::INITIALIZER, lazy) {}
    };

    struct NoopOp : public InitializerOp
    {
    public:
        NoopOp(LazyArrayPtr lazy) : InitializerOp(Opcode::NOOP, lazy) {}
    };

    struct ArangeOp : public InitializerOp
    {
    private:
        ShapeView view;
        isize start;
        isize step;
        DtypePtr dtype;

    public:
        ArangeOp(LazyArrayPtr lazy, const ShapeView &view, isize start, isize step, DtypePtr dtype) : InitializerOp(Opcode::ARANGE, lazy), view(view), start(start), step(step), dtype(dtype) {}
        const ShapeView &get_view() const { return view; }
        isize get_start() const { return start; }
        isize get_step() const { return step; }
        DtypePtr get_dtype() const { return dtype; }
        const std::string str() const override
        {
            return InitializerOp::str() + ", dtype: " + dtype->str() + ", view: (" + vnumstr(view) + "), start: " + std::to_string(start) + ", step: " + std::to_string(step);
        }
    };

    struct FullOp : public InitializerOp
    {
    private:
        ShapeView view;
        int c;
        DtypePtr dtype;

    public:
        FullOp(LazyArrayPtr lazy, const ShapeView &view, int c, DtypePtr dtype) : InitializerOp(Opcode::FULL, lazy), view(view), c(c), dtype(dtype) {}
        const ShapeView &get_view() const { return view; }
        int get_const() const { return c; }
        DtypePtr get_dtype() const { return dtype; }
        const std::string str() const override
        {
            auto s = InitializerOp::str() + ", dtype: " + dtype->str() + ", view: (" + vnumstr(view) + "), value: ";
            return s + dtype->get_value_as_str(c);
        }
    };

    struct BuffOp : public InitializerOp
    {
    public:
        BuffOp(LazyArrayPtr lazy) : InitializerOp(Opcode::BUFF, lazy) {}
        const std::string str() const override { return InitializerOp::str(); }
    };

    struct UnaryOp : public Op
    {
    protected:
        bool in_place;
        OpPtr operand;

    public:
        UnaryOp(Opcode opcode, LazyArrayPtr lazy, OpPtr operand, bool in_place) : Op(opcode, Optype::UNARY, lazy), operand(operand), in_place(in_place) {}
        OpPtr get_operand() const { return operand; }
        const std::string str() const override;
        bool is_in_place() const { return in_place; }
    };

    struct BinaryOp : public Op
    {
    protected:
        bool in_place;
        OpPtr lhs;
        OpPtr rhs;

    public:
        BinaryOp(Opcode opcode, LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : Op(opcode, Optype::BINARY, lazy), lhs(lhs), rhs(rhs), in_place(in_place) {}
        OpPtr get_lhs() const { return lhs; }
        OpPtr get_rhs() const { return rhs; }
        const std::string str() const override;
        bool is_in_place() const { return in_place; }
    };

    struct TransformOp : public Op
    {
    protected:
        OpPtr operand;

    public:
        TransformOp(Opcode opcode, LazyArrayPtr lazy, OpPtr operand) : Op(opcode, Optype::TRANSFORM, lazy), operand(operand) {}
        OpPtr get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct ReduceOp : public Op
    {
    protected:
        ReduceMode mode;
        OpPtr operand;
        ShapeDims dims;
        int default_val;

    public:
        ReduceOp(Opcode opcode, ReduceMode mode, LazyArrayPtr lazy, OpPtr operand, const ShapeDims &dims, int default_val) : Op(opcode, Optype::REDUCE, lazy), mode(mode), operand(operand), dims(dims), default_val(default_val) {}
        ReduceMode get_mode() const { return mode; }
        OpPtr get_operand() const { return operand; }
        const ShapeDims &get_dims() const { return dims; }
        int get_default_val() const { return default_val; }
        const std::string str() const override;
    };

    struct AddOp : public BinaryOp
    {
    public:
        AddOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::ADD, lazy, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct SubOp : public BinaryOp
    {
    public:
        SubOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::SUB, lazy, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct MulOp : public BinaryOp
    {
    public:
        MulOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::MUL, lazy, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct DivOp : public BinaryOp
    {
    public:
        DivOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::DIV, lazy, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct EqOp : public BinaryOp
    {
    public:
        EqOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::EQ, lazy, lhs, rhs, false) {}
    };

    struct NeqOp : public BinaryOp
    {
    public:
        NeqOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::NEQ, lazy, lhs, rhs, false) {}
    };

    struct LtOp : public BinaryOp
    {
    public:
        LtOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::LT, lazy, lhs, rhs, false) {}
    };

    struct GtOp : public BinaryOp
    {
    public:
        GtOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::GT, lazy, lhs, rhs, false) {}
    };

    struct LeqOp : public BinaryOp
    {
    public:
        LeqOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::LEQ, lazy, lhs, rhs, false) {}
    };

    struct GeqOp : public BinaryOp
    {
    public:
        GeqOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::GEQ, lazy, lhs, rhs, false) {}
    };

    struct MatmulOp : public Op
    {
    private:
        OpPtr lhs;
        OpPtr rhs;

    public:
        MatmulOp(LazyArrayPtr lazy, OpPtr lhs, OpPtr rhs) : Op(Opcode::MATMUL, Optype::MATMUL, lazy), lhs(lhs), rhs(rhs) {}
        OpPtr get_lhs() const { return lhs; }
        OpPtr get_rhs() const { return rhs; }
        const std::string str() const override;
        void backward() const override;
    };

    struct SqOp : public UnaryOp
    {
    public:
        SqOp(LazyArrayPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::SQ, lazy, operand, in_place) {}
        void backward() const override;
    };

    struct SqrtOp : public UnaryOp
    {
    public:
        SqrtOp(LazyArrayPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::SQRT, lazy, operand, in_place) {}
        void backward() const override;
    };

    struct NegOp : public UnaryOp
    {
    public:
        NegOp(LazyArrayPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::NEG, lazy, operand, in_place) {}
        void backward() const override;
    };

    struct IdentityOp : public UnaryOp
    {
    public:
        IdentityOp(LazyArrayPtr lazy, OpPtr operand) : UnaryOp(Opcode::IDENTITY, lazy, operand, false) {}
        void backward() const override;
    };

    struct ExpOp : public UnaryOp
    {
    public:
        ExpOp(LazyArrayPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::EXP, lazy, operand, in_place) {}
        void backward() const override;
    };

    struct LogOp : public UnaryOp
    {
    public:
        LogOp(LazyArrayPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::LOG, lazy, operand, in_place) {}
        void backward() const override;
    };

    struct RecipOp : public UnaryOp
    {
    public:
        RecipOp(LazyArrayPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::RECIP, lazy, operand, in_place) {}
        void backward() const override;
    };

    struct ReshapeOp : public TransformOp
    {
    private:
        ShapeView view;

    public:
        ReshapeOp(LazyArrayPtr lazy, OpPtr operand, const ShapeView &view) : TransformOp(Opcode::RESHAPE, lazy, operand), view(view) {}
        const ShapeView &get_view() const { return view; }
        const std::string str() const override { return TransformOp::str() + ", view: (" + vnumstr(view) + ")"; }
        void backward() const override;
    };

    struct SliceOp : public TransformOp
    {
    private:
        std::vector<Range> ranges;

    public:
        SliceOp(LazyArrayPtr lazy, OpPtr operand, const std::vector<Range> &ranges) : TransformOp(Opcode::SLICE, lazy, operand), ranges(ranges) {}
        const std::vector<Range> &get_ranges() const { return ranges; }
        const std::string str() const override
        {
            return TransformOp::str() + ", ranges:(" + vstr<Range>(ranges, [](Range range)
                                                                   { return range.str(); }) +
                   ")";
        }
        void backward() const override;
    };

    struct PermuteOp : public TransformOp
    {
    private:
        ShapeDims dims;

    public:
        PermuteOp(LazyArrayPtr lazy, OpPtr operand, const ShapeDims &dims) : TransformOp(Opcode::PERMUTE, lazy, operand), dims(dims) {}
        const ShapeDims &get_perm() const { return dims; }
        const std::string str() const override { return TransformOp::str() + ", permutation: (" + vnumstr(dims) + ")"; }
        void backward() const override;
    };

    struct BroadcastOp : public TransformOp
    {
    private:
        ShapeView input_view;
        ShapeView output_view;
        ShapeDims dims;

    public:
        BroadcastOp(LazyArrayPtr lazy, OpPtr operand, const ShapeView &input_view, const ShapeView &output_view, const ShapeDims &dims) : TransformOp(Opcode::BROADCAST, lazy, operand), input_view(input_view), output_view(output_view), dims(dims) {}
        const ShapeView &get_input_view() const { return input_view; }
        const ShapeView &get_output_view() const { return output_view; }
        const ShapeDims &get_dims() const { return dims; }
        const std::string str() const override { return TransformOp::str() + ", output view: (" + vnumstr(output_view) + ")"; }
        void backward() const override;
    };

    struct SqueezeOp : public TransformOp
    {
    private:
        isize dim;

    public:
        SqueezeOp(LazyArrayPtr lazy, OpPtr operand, isize dim) : TransformOp(Opcode::SQUEEZE, lazy, operand), dim(dim) {}
        isize get_dim() const { return dim; }
        const std::string str() const override { return TransformOp::str() + ", dim: " + std::to_string(dim); }
        void backward() const override;
    };

    struct UnsqueezeOp : public TransformOp
    {
    private:
        isize dim;

    public:
        UnsqueezeOp(LazyArrayPtr lazy, OpPtr operand, isize dim) : TransformOp(Opcode::UNSQUEEZE, lazy, operand), dim(dim) {}
        isize get_dim() const { return dim; }
        const std::string str() const override { return TransformOp::str() + ", dim: " + std::to_string(dim); }
        void backward() const override;
    };

    struct AstypeOp : public TransformOp
    {
    private:
        DtypePtr dtype;

    public:
        AstypeOp(LazyArrayPtr lazy, OpPtr operand, DtypePtr dtype) : TransformOp(Opcode::ASTYPE, lazy, operand), dtype(dtype) {}
        DtypePtr get_dtype() const { return dtype; }
        const std::string str() const override { return TransformOp::str() + ", dtype: " + dtype->str(); }
    };

    struct SumOp : public ReduceOp
    {
    public:
        SumOp(LazyArrayPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::SUM, ReduceMode::VALUE, lazy, operand, dims, 0) {}
        void backward() const override;
    };

    struct MaxOp : public ReduceOp
    {
    public:
        MaxOp(LazyArrayPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::MAX, ReduceMode::VALUE, lazy, operand, dims, lazy->get_dtype()->min()) {}
        void backward() const override;
    };

    struct MinOp : public ReduceOp
    {
    public:
        MinOp(LazyArrayPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::MIN, ReduceMode::VALUE, lazy, operand, dims, lazy->get_dtype()->max()) {}
        void backward() const override;
    };

    struct ArgmaxOp : public ReduceOp
    {
    public:
        ArgmaxOp(LazyArrayPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::ARGMAX, ReduceMode::ARG, lazy, operand, dims, operand->get_lazy()->get_dtype()->min()) {}
    };

    struct ArgminOp : public ReduceOp
    {
    public:
        ArgminOp(LazyArrayPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::ARGMIN, ReduceMode::ARG, lazy, operand, dims, operand->get_lazy()->get_dtype()->max()) {}
    };

    OpPtr detach(OpPtr op);
    OpPtr full_impl(const ShapeView &view, int c, DtypePtr dtype, DevicePtr device);
    OpPtr zeros(const ShapeView &view, DtypePtr dtype, DevicePtr device);
    OpPtr zeros_like(OpPtr in_op, DtypePtr dtype, DevicePtr device);
    OpPtr ones(const ShapeView &view, DtypePtr dtype, DevicePtr device);
    OpPtr ones_like(OpPtr in_op, DtypePtr dtype, DevicePtr device);
    OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype, DevicePtr device);
    OpPtr from_ptr(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype, DevicePtr device);
    OpPtr broadcast(OpPtr op, const ShapeView &view);
    OpPtr broadcast_to(OpPtr op, const ShapeView &view);
    OpPtr slice(OpPtr in_op, const RangeVec &ranges);
    OpPtr astype(OpPtr in_op, DtypePtr dtype);
    OpPtr unsqueeze(OpPtr in_op, isize dim);
    OpPtr squeeze(OpPtr in_op, isize dim);
    OpPtr add(OpPtr lop, OpPtr rop);
    OpPtr sub(OpPtr lop, OpPtr rop);
    OpPtr mul(OpPtr lop, OpPtr rop);
    OpPtr div(OpPtr lop, OpPtr rop);
    OpPtr matmul(OpPtr lop, OpPtr rop);
    OpPtr self_add(OpPtr lop, OpPtr rop);
    OpPtr self_sub(OpPtr lop, OpPtr rop);
    OpPtr self_mul(OpPtr lop, OpPtr rop);
    OpPtr self_div(OpPtr lop, OpPtr rop);
    OpPtr eq(OpPtr lop, OpPtr rop);
    OpPtr neq(OpPtr lop, OpPtr rop);
    OpPtr lt(OpPtr lop, OpPtr rop);
    OpPtr gt(OpPtr lop, OpPtr rop);
    OpPtr leq(OpPtr lop, OpPtr rop);
    OpPtr geq(OpPtr lop, OpPtr rop);
    OpPtr sq(OpPtr in_op, bool in_place = false);
    OpPtr sqrt(OpPtr in_op, bool in_place = false);
    OpPtr neg(OpPtr in_op, bool in_place = false);
    OpPtr identity(OpPtr in_op, bool in_place = false);
    OpPtr exp(OpPtr in_op, bool in_place = false);
    OpPtr log(OpPtr in_op, bool in_place = false);
    OpPtr recip(OpPtr in_op, bool in_place = false);
    OpPtr reshape(OpPtr in_op, const ShapeView &view);
    OpPtr permute(OpPtr in_op, const ShapeDims &dims);
    OpPtr transpose(OpPtr in_op, isize start_dim, isize end_dim);
    OpPtr flatten(OpPtr in_op, isize start_dim, isize end_dim);
    OpPtr sum(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr max(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr min(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr argmax(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr argmin(OpPtr in_op, const ShapeDims &dims = {});

    template <class T>
    void if_scalar_is_numeric(T c)
    {
        static_assert(std::is_arithmetic_v<T>, "Scalar type must be numeric");
    }

    template <class T>
    OpPtr full(const ShapeView &view, T c, DtypePtr dtype, DevicePtr device)
    {
        if_scalar_is_numeric(c);
        return full_impl(view, dtype_cast(c, dtype), dtype, device);
    }

    template <class T>
    OpPtr full_like(OpPtr in_op, T c, DtypePtr dtype, DevicePtr device)
    {
        return full(in_op->get_lazy()->get_view(), c, dtype, device);
    }

    template <class T>
    OpPtr add(OpPtr lop, T c)
    {
        LazyArrayPtr larr = lop->get_lazy();
        DtypePtr ldtype = larr->get_dtype();
        OpPtr rop = full(larr->get_view(), c, ldtype, larr->get_device());
        return add(lop, rop);
    }

    template <class T>
    OpPtr sub(OpPtr lop, T c)
    {
        LazyArrayPtr larr = lop->get_lazy();
        DtypePtr ldtype = larr->get_dtype();
        OpPtr rop = full(larr->get_view(), c, ldtype, larr->get_device());
        return sub(lop, rop);
    }

    template <class T>
    OpPtr mul(OpPtr lop, T c)
    {
        LazyArrayPtr larr = lop->get_lazy();
        DtypePtr ldtype = larr->get_dtype();
        OpPtr rop = full(larr->get_view(), c, ldtype, larr->get_device());
        return mul(lop, rop);
    }

    template <class T>
    OpPtr div(OpPtr lop, T c)
    {
        LazyArrayPtr larr = lop->get_lazy();
        DtypePtr ldtype = larr->get_dtype();
        OpPtr rop = full(larr->get_view(), c, ldtype, larr->get_device());
        return div(lop, rop);
    }

    template <class O>
    OpPtr binary_ss(OpPtr lop, OpPtr rop)
    {
        O dummy_op(nullptr, nullptr, nullptr, false);
        LazyArrayPtr larr = lop->get_lazy();
        LazyArrayPtr rarr = rop->get_lazy();
        const ShapeView &lview = larr->get_view();
        const ShapeView &rview = rarr->get_view();
        DtypePtr ldtype = larr->get_dtype();
        DtypePtr rdtype = rarr->get_dtype();
        DevicePtr ldevice = larr->get_device();
        DevicePtr rdevice = rarr->get_device();

        if (!larr->get_shape().broadcastable(rview))
        {
            throw IncompatShapesForOp(dummy_op.get_opcode_str(), vnumstr(lview), vnumstr(rview));
        }
        if (!binary_dtypes.contains(ldtype) || ldtype != rdtype)
        {
            throw IncompatDtypesForOp(dummy_op.get_opcode_str(), ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice)
        {
            throw IncompatDevicesForOp(dummy_op.get_opcode_str(), ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_lop = broadcast(lop, rview);
        OpPtr broadcasted_rop = broadcast(rop, lview);
        LazyArrayPtr out_arr = LazyArray::empty(Shape(broadcasted_lop->get_lazy()->get_view()), ldtype, ldevice);
        OpPtr out_op = std::make_shared<O>(out_arr, broadcasted_lop, broadcasted_rop, false);
        return out_op;
    }

    template <class O>
    OpPtr self_binary_ss(OpPtr lop, OpPtr rop)
    {
        O dummy_op(nullptr, nullptr, nullptr, true);
        LazyArrayPtr larr = lop->get_lazy();
        LazyArrayPtr rarr = rop->get_lazy();
        const Shape &lshape = larr->get_shape();
        const ShapeView &lview = larr->get_view();
        const ShapeView &rview = rarr->get_view();
        DtypePtr ldtype = larr->get_dtype();
        DtypePtr rdtype = rarr->get_dtype();
        DevicePtr ldevice = larr->get_device();
        DevicePtr rdevice = rarr->get_device();

        if (!rarr->get_shape().broadcastable_to(lview))
        {
            throw IncompatShapesForOp(dummy_op.get_opcode_str(), vnumstr(lview), vnumstr(rview));
        }
        if (!binary_dtypes.contains(ldtype) || ldtype != rdtype)
        {
            throw IncompatDtypesForOp(dummy_op.get_opcode_str(), ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice)
        {
            throw IncompatDevicesForOp(dummy_op.get_opcode_str(), ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_rop = broadcast_to(rop, lview);
        LazyArrayPtr out_arr = LazyArray::empty(lshape, ldtype, ldevice);
        OpPtr out_op = std::make_shared<O>(out_arr, lop, broadcasted_rop, true);
        return out_op;
    }

    template <class O>
    OpPtr unary_ss(OpPtr in_op, bool in_place)
    {
        LazyArrayPtr in_arr = in_op->get_lazy();
        DtypePtr in_dtype = in_arr->get_dtype();

        if (!float_dtype_by_dtype.contains(in_dtype))
        {
            O dummy_op(nullptr, nullptr, in_place);
            throw IncompatDtypeForOp(dummy_op.get_opcode_str(), in_dtype->str());
        }

        LazyArrayPtr out_arr = LazyArray::empty(Shape(in_arr->get_view()), in_dtype, in_arr->get_device());
        OpPtr out_op = std::make_shared<O>(out_arr, in_op, in_place);
        return out_op;
    }

    template <class O>
    OpPtr unary_ss_float(OpPtr in_op, bool in_place)
    {
        O dummy_op(nullptr, nullptr, in_place);
        LazyArrayPtr in_arr = in_op->get_lazy();
        DtypePtr in_dtype = in_arr->get_dtype();

        if (in_place)
        {
            if (in_dtype->get_type() != DtypeType::FLOAT)
            {
                // This method requires the operand to be of floating-point type
                // to do in-place operation since the result is of floating-point type
                throw IncompatDtypeForOp(dummy_op.get_opcode_str(), in_dtype->str());
            }
        }

        auto result_dtype = float_dtype_by_dtype.find(in_dtype);
        if (result_dtype == float_dtype_by_dtype.end())
        {
            throw IncompatDtypeForOp(dummy_op.get_opcode_str(), in_dtype->str());
        }

        LazyArrayPtr out_arr = LazyArray::empty(Shape(in_arr->get_view()), result_dtype->second, in_arr->get_device());
        OpPtr out_op = std::make_shared<O>(out_arr, in_op, in_place);
        return out_op;
    }

    template <class O>
    OpPtr cmp(OpPtr lop, OpPtr rop, DtypePtrSet &valid_dtypes)
    {
        O dummy_op(nullptr, nullptr, nullptr);
        LazyArrayPtr larr = lop->get_lazy();
        LazyArrayPtr rarr = rop->get_lazy();
        const ShapeView &lview = larr->get_view();
        const ShapeView &rview = rarr->get_view();
        DtypePtr ldtype = larr->get_dtype();
        DtypePtr rdtype = rarr->get_dtype();
        DevicePtr ldevice = larr->get_device();
        DevicePtr rdevice = rarr->get_device();

        if (!larr->get_shape().broadcastable(rview))
        {
            throw IncompatShapesForOp(dummy_op.get_opcode_str(), vnumstr(lview), vnumstr(rview));
        }
        if (!valid_dtypes.contains(ldtype) || ldtype != rdtype)
        {
            throw IncompatDtypesForOp(dummy_op.get_opcode_str(), ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice)
        {
            throw IncompatDevicesForOp(dummy_op.get_opcode_str(), ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_lop = broadcast(lop, rview);
        OpPtr broadcasted_rop = broadcast(rop, lview);
        LazyArrayPtr out_arr = LazyArray::empty(Shape(broadcasted_lop->get_lazy()->get_view()), &b8, ldevice);
        OpPtr out_op = std::make_shared<O>(out_arr, broadcasted_lop, broadcasted_rop);
        return out_op;
    }

    template <class O>
    OpPtr reduce(OpPtr in_op, const ShapeDims &dims, DtypePtr result_dtype, DtypePtrSet &valid_dtypes)
    {
        LazyArrayPtr in_arr = in_op->get_lazy();
        const Shape &in_shape = in_arr->get_shape();
        DtypePtr in_dtype = in_arr->get_dtype();
        DevicePtr in_device = in_arr->get_device();

        if (!valid_dtypes.contains(in_dtype))
        {
            O dummy_op(nullptr, nullptr, dims);
            throw IncompatDtypeForOp(dummy_op.get_opcode_str(), in_dtype->str());
        }

        LazyArrayPtr reduction_arr;
        OpPtr reduction_op;

        if (dims.size() == 0)
        {
            // Reduce to one element
            reduction_arr = LazyArray::empty(Shape({1}), result_dtype, in_device);
            reduction_op = std::make_shared<O>(reduction_arr, in_op, dims);
            return reduction_op;
        }

        // Fill kept_dims from 0 to ndim-1
        ShapeDims kept_dims(in_shape.get_ndim());
        ShapeDims reduction_dims;
        std::iota(kept_dims.begin(), kept_dims.end(), 0);

        // Remove the dimensions to be reduced from kept_dims and add them to reduction_dims
        for (auto &dim : dims)
        {
            auto iter = std::find(kept_dims.begin(), kept_dims.end(), dim);
            if (iter == kept_dims.end())
            {
                throw std::invalid_argument("Invalid reduction dimension " + std::to_string(dim) + " on array " + in_arr->get_id().str() + ".");
            }
            else
            {
                kept_dims.erase(iter);
                reduction_dims.emplace_back(dim);
            }
        }

        // Permute the array by moving all reduced dimensions to the end of the view
        ShapeDims permutation_dims;
        permutation_dims.insert(permutation_dims.end(), kept_dims.begin(), kept_dims.end());
        permutation_dims.insert(permutation_dims.end(), reduction_dims.begin(), reduction_dims.end());
        OpPtr permutation_op = permute(in_op, permutation_dims);

        // Reshape the array to 2D for reduction
        ShapeView kept_view;
        isize kept_numel = 1;
        isize reduction_numel = 1;

        for (auto &dim : kept_dims)
        {
            kept_view.emplace_back(in_shape[dim]);
            kept_numel *= in_shape[dim];
        }
        for (auto &dim : reduction_dims)
        {
            reduction_numel *= in_shape[dim];
        }

        OpPtr reshape_op_before_reduction = reshape(permutation_op, {kept_numel, reduction_numel});
        // Reduce the array
        reduction_arr = LazyArray::empty(Shape({kept_numel, 1}), result_dtype, in_device);
        reduction_op = std::make_shared<O>(reduction_arr, reshape_op_before_reduction, dims);
        // Reshape the array back to the shape without reduced dimensions(except for 1 at the end)
        kept_view.emplace_back(1);
        OpPtr reshape_op_after_reduction = reshape(reduction_op, kept_view);
        return reshape_op_after_reduction;
    }
}