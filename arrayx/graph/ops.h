#pragma once

#include "../utils.h"
#include "../core/array.h"

namespace ax::graph
{
    using namespace ax::core;
    enum class Opcode
    {
        NOOP,
        RANDN,
        ARANGE,
        FULL,
        BUFF,
        NUMPY,
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
        AS_TYPE
    };

    enum class Optype
    {
        INITIALIZER,
        UNARY,
        BINARY,
        MATMUL,
        TRANSFORM,
        REDUCE
    };

    enum class ReduceMode
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
        {Opcode::NUMPY, "numpy"},
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
        ArrayPtr output;
        std::shared_ptr<Op> grad = nullptr;
        std::shared_ptr<Op> gradroot = nullptr;

    public:
        Op(Opcode opcode, Optype optype, ArrayPtr output) : opcode(opcode), optype(optype), output(output) {}
        Op(const Op &) = delete;
        Op &operator=(const Op &) = delete;
        virtual ~Op() = default;
        Opcode get_opcode() const { return opcode; }
        const std::string &get_opcode_str() const { return str_by_opname.at(opcode); }
        Optype get_optype() const { return optype; }
        ArrayPtr get_output() const { return output; }
        std::shared_ptr<Op> get_grad() const { return grad; }
        std::shared_ptr<Op> get_gradroot() const { return gradroot; }
        virtual void backward() const {}
        void init_grad(bool with_zeros = true);
        void update_grad(std::shared_ptr<Op> grad, bool sub = false);
        const std::string str() const override
        {
            return output->get_id().str() + ": opcode: " + get_opcode_str();
        }
    };

    using OpPtr = std::shared_ptr<Op>;

    struct InitializerOp : public Op
    {
    public:
        InitializerOp(Opcode opcode, ArrayPtr output) : Op(opcode, Optype::INITIALIZER, output) {}
    };

    struct NoopOp : public InitializerOp
    {
    public:
        NoopOp(ArrayPtr output) : InitializerOp(Opcode::NOOP, output) {}
    };

    struct ArangeOp : public InitializerOp
    {
    private:
        ShapeView view;
        isize start;
        isize step;
        DtypePtr dtype;

    public:
        ArangeOp(ArrayPtr output, const ShapeView &view, isize start, isize step, DtypePtr dtype) : InitializerOp(Opcode::ARANGE, output), view(view), start(start), step(step), dtype(dtype) {}
        const ShapeView &get_view() { return view; }
        isize get_start() { return start; }
        isize get_step() { return step; }
        DtypePtr get_dtype() { return dtype; }
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
        FullOp(ArrayPtr output, const ShapeView &view, int c, DtypePtr dtype) : InitializerOp(Opcode::FULL, output), view(view), c(c), dtype(dtype) {}
        const ShapeView &get_view() { return view; }
        int get_const() const { return c; }
        DtypePtr get_dtype() { return dtype; }
        const std::string str() const override
        {
            auto s = InitializerOp::str() + ", dtype: " + dtype->str() + ", view: (" + vnumstr(view) + "), value: ";
            return s + dtype->get_value_as_str(c);
        }
    };

    struct BuffOp : public InitializerOp
    {
    public:
        BuffOp(ArrayPtr output) : InitializerOp(Opcode::BUFF, output) {}
        const std::string str() const override { return InitializerOp::str(); }
    };

    struct NumpyOp : public InitializerOp
    {
    public:
        NumpyOp(ArrayPtr output) : InitializerOp(Opcode::NUMPY, output) {}
        const std::string str() const override { return InitializerOp::str(); }
    };

    struct UnaryOp : public Op
    {
    protected:
        bool in_place;
        OpPtr operand;

    public:
        UnaryOp(Opcode opcode, ArrayPtr output, OpPtr operand, bool in_place) : Op(opcode, Optype::UNARY, output), operand(operand), in_place(in_place) {}
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
        BinaryOp(Opcode opcode, ArrayPtr output, OpPtr lhs, OpPtr rhs, bool in_place) : Op(opcode, Optype::BINARY, output), lhs(lhs), rhs(rhs), in_place(in_place) {}
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
        TransformOp(Opcode opcode, ArrayPtr output, OpPtr operand) : Op(opcode, Optype::TRANSFORM, output), operand(operand) {}
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
        ReduceOp(Opcode opcode, ReduceMode mode, ArrayPtr output, OpPtr operand, const ShapeDims &dims, int default_val) : Op(opcode, Optype::REDUCE, output), mode(mode), operand(operand), dims(dims), default_val(default_val) {}
        ReduceMode get_mode() const { return mode; }
        OpPtr get_operand() const { return operand; }
        const ShapeDims &get_dims() { return dims; }
        int get_default_val() const { return default_val; }
        const std::string str() const override;
    };

    struct AddOp : public BinaryOp
    {
    public:
        AddOp(ArrayPtr output, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::ADD, output, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct SubOp : public BinaryOp
    {
    public:
        SubOp(ArrayPtr output, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::SUB, output, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct MulOp : public BinaryOp
    {
    public:
        MulOp(ArrayPtr output, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::MUL, output, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct DivOp : public BinaryOp
    {
    public:
        DivOp(ArrayPtr output, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(Opcode::DIV, output, lhs, rhs, in_place) {}

        void backward() const override;
    };

    struct EqOp : public BinaryOp
    {
    public:
        EqOp(ArrayPtr output, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::EQ, output, lhs, rhs, false) {}
    };

    struct NeqOp : public BinaryOp
    {
    public:
        NeqOp(ArrayPtr output, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::NEQ, output, lhs, rhs, false) {}
    };

    struct LtOp : public BinaryOp
    {
    public:
        LtOp(ArrayPtr output, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::LT, output, lhs, rhs, false) {}
    };

    struct GtOp : public BinaryOp
    {
    public:
        GtOp(ArrayPtr output, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::GT, output, lhs, rhs, false) {}
    };

    struct LeqOp : public BinaryOp
    {
    public:
        LeqOp(ArrayPtr output, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::LEQ, output, lhs, rhs, false) {}
    };

    struct GeqOp : public BinaryOp
    {
    public:
        GeqOp(ArrayPtr output, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::GEQ, output, lhs, rhs, false) {}
    };

    struct MatmulOp : public Op
    {
    private:
        OpPtr lhs;
        OpPtr rhs;

    public:
        MatmulOp(ArrayPtr output, OpPtr lhs, OpPtr rhs) : Op(Opcode::MATMUL, Optype::MATMUL, output), lhs(lhs), rhs(rhs) {}
        OpPtr get_lhs() const { return lhs; }
        OpPtr get_rhs() const { return rhs; }
        const std::string str() const override;
        // void backward() const override;
    };

    struct SqOp : public UnaryOp
    {
    public:
        SqOp(ArrayPtr output, OpPtr operand, bool in_place) : UnaryOp(Opcode::SQ, output, operand, in_place) {}
        // void backward() const override;
    };

    struct SqrtOp : public UnaryOp
    {
    public:
        SqrtOp(ArrayPtr output, OpPtr operand, bool in_place) : UnaryOp(Opcode::SQRT, output, operand, in_place) {}
        // void backward() const override;
    };

    struct NegOp : public UnaryOp
    {
    public:
        NegOp(ArrayPtr output, OpPtr operand, bool in_place) : UnaryOp(Opcode::NEG, output, operand, in_place) {}
        // void backward() const override;
    };

    struct IdentityOp : public UnaryOp
    {
    public:
        IdentityOp(ArrayPtr output, OpPtr operand) : UnaryOp(Opcode::IDENTITY, output, operand, false) {}
        // void backward() const override;
    };

    struct ExpOp : public UnaryOp
    {
    public:
        ExpOp(ArrayPtr output, OpPtr operand, bool in_place) : UnaryOp(Opcode::EXP, output, operand, in_place) {}
        // void backward() const override;
    };

    struct LogOp : public UnaryOp
    {
    public:
        LogOp(ArrayPtr output, OpPtr operand, bool in_place) : UnaryOp(Opcode::LOG, output, operand, in_place) {}
        // void backward() const override;
    };

    struct RecipOp : public UnaryOp
    {
    public:
        RecipOp(ArrayPtr output, OpPtr operand, bool in_place) : UnaryOp(Opcode::RECIP, output, operand, in_place) {}
        // void backward() const override;
    };

    struct ReshapeOp : public TransformOp
    {
    private:
        ShapeView view;

    public:
        ReshapeOp(ArrayPtr output, OpPtr operand, const ShapeView &view) : TransformOp(Opcode::RESHAPE, output, operand), view(view) {}
        const ShapeView &get_view() { return view; }
        const std::string str() const override { return str() + ", view: (" + vnumstr(view) + ")"; }
        // void backward() const override;
    };

    struct SliceOp : public TransformOp
    {
    private:
        std::vector<Range> ranges;

    public:
        SliceOp(ArrayPtr output, OpPtr operand, const std::vector<Range> &ranges) : TransformOp(Opcode::SLICE, output, operand), ranges(ranges) {}
        const std::vector<Range> &get_ranges() { return ranges; }
        const std::string str() const override
        {
            return str() + ", ranges:(" + vstr<Range>(ranges, [](Range range)
                                                      { return range.str(); }) +
                   ")";
        }
        // void backward() const override;
    };

    struct PermuteOp : public TransformOp
    {
    private:
        ShapeDims dims;

    public:
        PermuteOp(ArrayPtr output, OpPtr operand, const ShapeDims &dims) : TransformOp(Opcode::PERMUTE, output, operand), dims(dims) {}
        const ShapeDims &get_perm() { return dims; }
        const std::string str() const override { return str() + ", permutation: (" + vnumstr(dims) + ")"; }
        // void backward() const override;
    };

    struct BroadcastOp : public TransformOp
    {
    private:
        ShapeView input_view;
        ShapeView output_view;
        ShapeDims dims;

    public:
        BroadcastOp(ArrayPtr output, OpPtr operand, const ShapeView &input_view, const ShapeView &output_view, const ShapeDims &dims) : TransformOp(Opcode::BROADCAST, output, operand), input_view(input_view), output_view(output_view), dims(dims) {}
        const ShapeView &get_input_view() { return input_view; }
        const ShapeView &get_output_view() { return output_view; }
        const ShapeDims &get_dims() { return dims; }
        const std::string str() const override { return str() + ", output view: (" + vnumstr(output_view) + ")"; }
        // void backward() const override;
    };

    struct SqueezeOp : public TransformOp
    {
    private:
        isize dim;

    public:
        SqueezeOp(ArrayPtr output, OpPtr operand, isize dim) : TransformOp(Opcode::SQUEEZE, output, operand), dim(dim) {}
        isize get_dim() { return dim; }
        const std::string str() const override { return str() + ", dim: " + std::to_string(dim); }
        // void backward() const override;
    };

    struct UnsqueezeOp : public TransformOp
    {
    private:
        isize dim;

    public:
        UnsqueezeOp(ArrayPtr output, OpPtr operand, isize dim) : TransformOp(Opcode::UNSQUEEZE, output, operand), dim(dim) {}
        isize get_dim() { return dim; }
        const std::string str() const override { return str() + ", dim: " + std::to_string(dim); }
        // void backward() const override;
    };

    struct AsTypeOp : public TransformOp
    {
    private:
        DtypePtr dtype;

    public:
        AsTypeOp(ArrayPtr output, OpPtr operand, DtypePtr dtype) : TransformOp(Opcode::AS_TYPE, output, operand), dtype(dtype) {}
        DtypePtr get_dtype() { return dtype; }
        const std::string str() const override { return str() + ", dtype: " + dtype->str(); }
    };

    struct SumOp : public ReduceOp
    {
    public:
        SumOp(ArrayPtr output, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::SUM, ReduceMode::VALUE, output, operand, dims, 0) {}
        // void backward() const override;
    };

    struct MaxOp : public ReduceOp
    {
    public:
        MaxOp(ArrayPtr output, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::MAX, ReduceMode::VALUE, output, operand, dims, output->get_dtype()->min()) {}
        // void backward() const override;
    };

    struct MinOp : public ReduceOp
    {
    public:
        MinOp(ArrayPtr output, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::MIN, ReduceMode::VALUE, output, operand, dims, output->get_dtype()->max()) {}
        // void backward() const override;
    };

    struct ArgmaxOp : public ReduceOp
    {
    public:
        ArgmaxOp(ArrayPtr output, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::ARGMAX, ReduceMode::ARG, output, operand, dims, operand->get_output()->get_dtype()->min()) {}
    };

    struct ArgminOp : public ReduceOp
    {
    public:
        ArgminOp(ArrayPtr output, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::ARGMIN, ReduceMode::ARG, output, operand, dims, operand->get_output()->get_dtype()->max()) {}
    };

    OpPtr detach(OpPtr op);
    OpPtr full(const ShapeView &view, int c, DtypePtr dtype = &f32, DevicePtr device = &device0);
    OpPtr zeros(const ShapeView &view, DtypePtr dtype = &f32, DevicePtr device = &device0);
    OpPtr ones(const ShapeView &view, DtypePtr dtype = &f32, DevicePtr device = &device0);
    OpPtr arange(const ShapeView &view, isize start, isize step, DtypePtr dtype = &f32, DevicePtr device = &device0);
    OpPtr from_buff(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype = &f32, DevicePtr device = &device0);
    OpPtr from_numpy(uint8_t *ptr, isize nbytes, const Shape &shape, DtypePtr dtype = &f32, DevicePtr device = &device0);
    OpPtr broadcast(OpPtr op, const ShapeView &view);
    OpPtr broadcast_to(OpPtr op, const ShapeView &view);
    OpPtr add(OpPtr lop, OpPtr rop);
    OpPtr sub(OpPtr lop, OpPtr rop);
    OpPtr mul(OpPtr lop, OpPtr rop);
    OpPtr div(OpPtr lop, OpPtr rop);
    OpPtr self_add(OpPtr lop, OpPtr rop);
    OpPtr self_sub(OpPtr lop, OpPtr rop);
    OpPtr self_mul(OpPtr lop, OpPtr rop);
    OpPtr self_div(OpPtr lop, OpPtr rop);

    template <class O>
    OpPtr binary_ss(OpPtr lop, OpPtr rop)
    {
        O dummy_op(nullptr, nullptr, nullptr, false);
        ArrayPtr larr = lop->get_output();
        ArrayPtr rarr = rop->get_output();
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
        if (!binary_dtypes.contains(ldtype) || !binary_dtypes.contains(rdtype) || ldtype != rdtype)
        {
            throw IncompatDtypesForOp(dummy_op.get_opcode_str(), ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice)
        {
            throw IncompatDevicesForOp(dummy_op.get_opcode_str(), ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_lop = broadcast(lop, rview);
        OpPtr broadcasted_rop = broadcast(rop, lview);
        ArrayPtr out_arr = Array::empty(Shape(broadcasted_lop->get_output()->get_view()), ldtype, ldevice);
        OpPtr out_op = std::make_shared<O>(out_arr, broadcasted_lop, broadcasted_rop, false);
        return out_op;
    }

    template <class O>
    OpPtr self_binary_ss(OpPtr lop, OpPtr rop)
    {
        O dummy_op(nullptr, nullptr, nullptr, true);
        ArrayPtr larr = lop->get_output();
        ArrayPtr rarr = rop->get_output();
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
        if (!binary_dtypes.contains(ldtype) || !binary_dtypes.contains(rdtype) || ldtype != rdtype)
        {
            throw IncompatDtypesForOp(dummy_op.get_opcode_str(), ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice)
        {
            throw IncompatDevicesForOp(dummy_op.get_opcode_str(), ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_rop = broadcast_to(rop, lview);
        ArrayPtr out_arr = Array::empty(lshape, ldtype, ldevice);
        OpPtr out_op = std::make_shared<O>(out_arr, lop, broadcasted_rop, true);
        return out_op;
    }
}