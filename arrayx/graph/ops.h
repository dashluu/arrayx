#pragma once

#include "../core/lazy_iter.h"
#include "../device/device.h"
#include "../utils.h"

namespace ax::graph {
    using namespace ax::core;
    using namespace ax::device;

    enum struct Opcode {
        NOP,
        RANDN,
        ARANGE,
        FULL,
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
        MINIMUM,
        MAXIMUM,
        MATMUL,
        SQ,
        SQRT,
        NEG,
        COPY,
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
        ASTYPE,
        // Used to get the number of enums
        COUNT
    };

    enum struct Optype {
        INITIALIZER,
        UNARY,
        BINARY,
        TRANSFORM,
        REDUCE
    };

    enum struct BinaryMode {
        ELMWISE,
        CMP,
        MATMUL
    };

    enum struct ReduceMode {
        VALUE,
        ARG,
    };

    struct Op;
    using OpPtr = std::shared_ptr<Op>;
    OpPtr detach(OpPtr op);

    struct Op : public std::enable_shared_from_this<Op> {
    protected:
        Opcode opcode;
        Optype optype;
        LazyPtr lazy;
        bool idempotent = true;
        // Note: grad_enabled cannot be used to set gradient flow
        // once the computational graph is compiled
        bool grad_enabled = true;

    public:
        OpPtr grad = nullptr;
        OpPtr grad_root = nullptr;

        Op(Opcode opcode, Optype optype, LazyPtr lazy) : opcode(opcode), optype(optype), lazy(lazy) {}
        Op(const Op &) = delete;
        Op &operator=(const Op &) = delete;
        virtual ~Op() = default;
        Opcode get_opcode() const { return opcode; }
        virtual const std::string &get_opname() const = 0;
        Optype get_optype() const { return optype; }
        LazyPtr get_lazy() const { return lazy; }
        OpPtr de_op() const { return detach(std::const_pointer_cast<Op>(shared_from_this())); }
        bool is_grad_enabled() const { return grad_enabled; }
        virtual void enable_grad(bool enabled) { grad_enabled = enabled; }
        bool is_idempotent() const { return idempotent; }
        virtual void backward() const {}
        void init_grad(bool with_zeros = true);
        void update_grad(OpPtr grad, bool sub = false);
        virtual const std::string str() const { return lazy->get_id().str() + ": opname: " + get_opname() + ", shape: " + lazy->get_shape().str() + ", dtype: " + lazy->get_dtype()->str(); }
    };

    struct InitializerOp : public Op {
    public:
        InitializerOp(Opcode opcode, LazyPtr lazy) : Op(opcode, Optype::INITIALIZER, lazy) {}
    };

    struct Nop : public InitializerOp {
    public:
        static constexpr std::string opname = "nop";
        Nop(LazyPtr lazy) : InitializerOp(Opcode::NOP, lazy) {}
        const std::string &get_opname() const override { return opname; }
    };

    struct ArangeOp : public InitializerOp {
    private:
        ShapeView view;
        isize start;
        isize step;
        DtypePtr dtype;

    public:
        static constexpr std::string opname = "arange";
        ArangeOp(LazyPtr lazy, const ShapeView &view, isize start, isize step, DtypePtr dtype) : InitializerOp(Opcode::ARANGE, lazy), view(view), start(start), step(step), dtype(dtype) {}
        const ShapeView &get_view() const { return view; }
        isize get_start() const { return start; }
        isize get_step() const { return step; }
        DtypePtr get_dtype() const { return dtype; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override { return InitializerOp::str() + ", dtype: " + dtype->str() + ", view: (" + vnumstr(view) + "), start: " + std::to_string(start) + ", step: " + std::to_string(step); }
    };

    struct FullOp : public InitializerOp {
    private:
        ShapeView view;
        isize c;
        DtypePtr dtype;

    public:
        static constexpr std::string opname = "full";
        FullOp(LazyPtr lazy, const ShapeView &view, isize c, DtypePtr dtype) : InitializerOp(Opcode::FULL, lazy), view(view), c(c), dtype(dtype) {}
        const ShapeView &get_view() const { return view; }
        isize get_const() const { return c; }
        DtypePtr get_dtype() const { return dtype; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override {
            auto s = InitializerOp::str() + ", dtype: " + dtype->str() + ", view: (" + vnumstr(view) + "), value: ";
            return s + dtype->get_value_as_str(c);
        }
    };

    struct UnaryOp : public Op {
    protected:
        bool in_place;
        OpPtr operand;

    public:
        UnaryOp(Opcode opcode, LazyPtr lazy, OpPtr operand, bool in_place) : Op(opcode, Optype::UNARY, lazy), operand(operand), in_place(in_place) {
            if (operand != nullptr) {
                idempotent = !in_place && operand->is_idempotent();
            }
        }
        OpPtr get_operand() const { return operand; }
        OpPtr de_operand() const { return detach(operand); }
        bool is_in_place() const { return in_place; }
        const std::string str() const override { return Op::str() + ", in-place: " + std::to_string(in_place) + ", operand: " + operand->get_lazy()->get_id().str(); }
    };

    struct BinaryOp : public Op {
    protected:
        OpPtr lhs;
        OpPtr rhs;
        BinaryMode mode;

    public:
        BinaryOp(Opcode opcode, BinaryMode mode, LazyPtr lazy, OpPtr lhs, OpPtr rhs) : Op(opcode, Optype::BINARY, lazy), mode(mode), lhs(lhs), rhs(rhs) {}
        BinaryMode get_mode() const { return mode; }
        OpPtr get_lhs() const { return lhs; }
        OpPtr get_rhs() const { return rhs; }
        OpPtr de_lhs() const { return detach(lhs); }
        OpPtr de_rhs() const { return detach(rhs); }
        const std::string str() const override { return Op::str() + ", lhs: " + lhs->get_lazy()->get_id().str() + ", rhs: " + rhs->get_lazy()->get_id().str(); }
    };

    struct ElmwiseBinaryOp : public BinaryOp {
    protected:
        bool in_place;

    public:
        ElmwiseBinaryOp(Opcode opcode, LazyPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : BinaryOp(opcode, BinaryMode::ELMWISE, lazy, lhs, rhs), in_place(in_place) {
            if (lhs != nullptr && rhs != nullptr) {
                idempotent = lhs->is_idempotent() && rhs->is_idempotent();
            }
        }

        bool is_in_place() const { return in_place; }
        const std::string str() const override { return Op::str() + ", in-place: " + std::to_string(in_place) + ", lhs: " + lhs->get_lazy()->get_id().str() + ", rhs: " + rhs->get_lazy()->get_id().str(); }
    };

    struct CmpOp : public BinaryOp {
    public:
        CmpOp(Opcode opcode, LazyPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(opcode, BinaryMode::CMP, lazy, lhs, rhs) { grad_enabled = false; }
        void enable_grad(bool enabled) override { grad_enabled = false; }
    };

    struct TransformOp : public Op {
    protected:
        OpPtr operand;

    public:
        TransformOp(Opcode opcode, LazyPtr lazy, OpPtr operand) : Op(opcode, Optype::TRANSFORM, lazy), operand(operand) {
            if (operand != nullptr) {
                idempotent = operand->is_idempotent();
            }
        }

        OpPtr get_operand() const { return operand; }
        OpPtr de_operand() const { return detach(operand); }
        const std::string str() const override { return Op::str() + ", operand: " + operand->get_lazy()->get_id().str(); }
    };

    struct ReduceOp : public Op {
    protected:
        ReduceMode mode;
        OpPtr operand;
        ShapeDims dims;
        isize default_val;

    public:
        ReduceOp(Opcode opcode, ReduceMode mode, LazyPtr lazy, OpPtr operand, const ShapeDims &dims, isize default_val) : Op(opcode, Optype::REDUCE, lazy), mode(mode), operand(operand), dims(dims), default_val(default_val) {
            if (operand != nullptr) {
                idempotent = operand->is_idempotent();
            }
        }

        ReduceMode get_mode() const { return mode; }
        OpPtr get_operand() const { return operand; }
        OpPtr de_operand() const { return detach(operand); }
        const ShapeDims &get_dims() const { return dims; }
        isize get_default_val() const { return default_val; }
        const std::string str() const override { return Op::str() + ", operand: " + operand->get_lazy()->get_id().str() + ", dims: " + vnumstr(dims) + ", default value: " + std::to_string(default_val); }
    };

    struct AddOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string opname = "add";
        AddOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(Opcode::ADD, lazy, lhs, rhs, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct SubOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string opname = "sub";
        SubOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(Opcode::SUB, lazy, lhs, rhs, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct MulOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string opname = "mul";
        MulOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(Opcode::MUL, lazy, lhs, rhs, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct DivOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string opname = "div";
        DivOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(Opcode::DIV, lazy, lhs, rhs, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct EqOp : public CmpOp {
    public:
        static constexpr std::string opname = "eq";
        EqOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs) : CmpOp(Opcode::EQ, lazy, lhs, rhs) {}
        const std::string &get_opname() const override { return opname; }
    };

    struct NeqOp : public CmpOp {
    public:
        static constexpr std::string opname = "neq";
        NeqOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs) : CmpOp(Opcode::NEQ, lazy, lhs, rhs) {}
        const std::string &get_opname() const override { return opname; }
    };

    struct LtOp : public CmpOp {
    public:
        static constexpr std::string opname = "lt";
        LtOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs) : CmpOp(Opcode::LT, lazy, lhs, rhs) {}
        const std::string &get_opname() const override { return opname; }
    };

    struct GtOp : public CmpOp {
    public:
        static constexpr std::string opname = "gt";
        GtOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs) : CmpOp(Opcode::GT, lazy, lhs, rhs) {}
        const std::string &get_opname() const override { return opname; }
    };

    struct LeqOp : public CmpOp {
    public:
        static constexpr std::string opname = "leq";
        LeqOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs) : CmpOp(Opcode::LEQ, lazy, lhs, rhs) {}
        const std::string &get_opname() const override { return opname; }
    };

    struct GeqOp : public CmpOp {
    public:
        static constexpr std::string opname = "geq";
        GeqOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs) : CmpOp(Opcode::GEQ, lazy, lhs, rhs) {}
        const std::string &get_opname() const override { return opname; }
    };

    struct MinimumOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string opname = "minimum";
        MinimumOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(Opcode::MINIMUM, lazy, lhs, rhs, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct MaximumOp : public ElmwiseBinaryOp {
    public:
        static constexpr std::string opname = "maximum";
        MaximumOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs, bool in_place) : ElmwiseBinaryOp(Opcode::MAXIMUM, lazy, lhs, rhs, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct MatmulOp : public BinaryOp {
    public:
        static constexpr std::string opname = "matmul";
        MatmulOp(LazyPtr lazy, OpPtr lhs, OpPtr rhs) : BinaryOp(Opcode::MATMUL, BinaryMode::MATMUL, lazy, lhs, rhs) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct SqOp : public UnaryOp {
    public:
        static constexpr std::string opname = "sq";
        SqOp(LazyPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::SQ, lazy, operand, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct SqrtOp : public UnaryOp {
    public:
        static constexpr std::string opname = "sqrt";
        SqrtOp(LazyPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::SQRT, lazy, operand, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct NegOp : public UnaryOp {
    public:
        static constexpr std::string opname = "neg";
        NegOp(LazyPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::NEG, lazy, operand, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct CopyOp : public UnaryOp {
    public:
        static constexpr std::string opname = "copy";
        CopyOp(LazyPtr lazy, OpPtr operand) : UnaryOp(Opcode::COPY, lazy, operand, false) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct ExpOp : public UnaryOp {
    public:
        static constexpr std::string opname = "exp";
        ExpOp(LazyPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::EXP, lazy, operand, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct LogOp : public UnaryOp {
    public:
        static constexpr std::string opname = "log";
        LogOp(LazyPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::LOG, lazy, operand, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct RecipOp : public UnaryOp {
    public:
        static constexpr std::string opname = "recip";
        RecipOp(LazyPtr lazy, OpPtr operand, bool in_place) : UnaryOp(Opcode::RECIP, lazy, operand, in_place) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct ReshapeOp : public TransformOp {
    private:
        ShapeView view;

    public:
        static constexpr std::string opname = "reshape";
        ReshapeOp(LazyPtr lazy, OpPtr operand, const ShapeView &view) : TransformOp(Opcode::RESHAPE, lazy, operand), view(view) {}
        const ShapeView &get_view() const { return view; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override { return TransformOp::str() + ", view: (" + vnumstr(view) + ")"; }
        void backward() const override;
    };

    struct SliceOp : public TransformOp {
    private:
        std::vector<Range> ranges;

    public:
        static constexpr std::string opname = "slice";
        SliceOp(LazyPtr lazy, OpPtr operand, const std::vector<Range> &ranges) : TransformOp(Opcode::SLICE, lazy, operand), ranges(ranges) {}
        const std::vector<Range> &get_ranges() const { return ranges; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override {
            return TransformOp::str() + ", ranges:(" + vstr<Range>(ranges, [](Range range) { return range.str(); }) + ")";
        }
        void backward() const override;
    };

    struct PermuteOp : public TransformOp {
    private:
        ShapeDims dims;

    public:
        static constexpr std::string opname = "permute";
        PermuteOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : TransformOp(Opcode::PERMUTE, lazy, operand), dims(dims) {}
        const ShapeDims &get_perm() const { return dims; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override { return TransformOp::str() + ", permutation: (" + vnumstr(dims) + ")"; }
        void backward() const override;
    };

    struct BroadcastOp : public TransformOp {
    private:
        ShapeView input_view;
        ShapeView output_view;
        ShapeDims dims;

    public:
        static constexpr std::string opname = "broadcast";
        BroadcastOp(LazyPtr lazy, OpPtr operand, const ShapeView &input_view, const ShapeView &output_view, const ShapeDims &dims) : TransformOp(Opcode::BROADCAST, lazy, operand), input_view(input_view), output_view(output_view), dims(dims) {}
        const ShapeView &get_input_view() const { return input_view; }
        const ShapeView &get_output_view() const { return output_view; }
        const ShapeDims &get_dims() const { return dims; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override { return TransformOp::str() + ", output view: (" + vnumstr(output_view) + ")"; }
        void backward() const override;
    };

    struct SqueezeOp : public TransformOp {
    private:
        ShapeDims dims;

    public:
        static constexpr std::string opname = "squeeze";
        SqueezeOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : TransformOp(Opcode::SQUEEZE, lazy, operand), dims(dims) {}
        const ShapeDims &get_dims() const { return dims; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override { return TransformOp::str() + ", dims: " + vnumstr(dims); }
        void backward() const override;
    };

    struct UnsqueezeOp : public TransformOp {
    private:
        ShapeDims dims;

    public:
        static constexpr std::string opname = "unsqueeze";
        UnsqueezeOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : TransformOp(Opcode::UNSQUEEZE, lazy, operand), dims(dims) {}
        const ShapeDims &get_dims() const { return dims; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override { return TransformOp::str() + ", dims: " + vnumstr(dims); }
        void backward() const override;
    };

    struct AstypeOp : public TransformOp {
    private:
        DtypePtr dtype;

    public:
        static constexpr std::string opname = "astype";
        AstypeOp(LazyPtr lazy, OpPtr operand, DtypePtr dtype) : TransformOp(Opcode::ASTYPE, lazy, operand), dtype(dtype) { grad_enabled = false; }
        void enable_grad(bool enabled) override { grad_enabled = false; }
        DtypePtr get_dtype() const { return dtype; }
        const std::string &get_opname() const override { return opname; }
        const std::string str() const override { return TransformOp::str() + ", dtype: " + dtype->str(); }
    };

    struct SumOp : public ReduceOp {
    public:
        static constexpr std::string opname = "sum";
        SumOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::SUM, ReduceMode::VALUE, lazy, operand, dims, 0) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct MaxOp : public ReduceOp {
    public:
        static constexpr std::string opname = "max";
        MaxOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::MAX, ReduceMode::VALUE, lazy, operand, dims, lazy->get_dtype()->min()) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct MinOp : public ReduceOp {
    public:
        static constexpr std::string opname = "min";
        MinOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::MIN, ReduceMode::VALUE, lazy, operand, dims, lazy->get_dtype()->max()) {}
        const std::string &get_opname() const override { return opname; }
        void backward() const override;
    };

    struct ArgmaxOp : public ReduceOp {
    public:
        static constexpr std::string opname = "argmax";
        ArgmaxOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::ARGMAX, ReduceMode::ARG, lazy, operand, dims, operand->get_lazy()->get_dtype()->min()) {
            grad_enabled = false;
        }
        void enable_grad(bool enabled) override { grad_enabled = false; }
        const std::string &get_opname() const override { return opname; }
    };

    struct ArgminOp : public ReduceOp {
    public:
        static constexpr std::string opname = "argmin";
        ArgminOp(LazyPtr lazy, OpPtr operand, const ShapeDims &dims) : ReduceOp(Opcode::ARGMIN, ReduceMode::ARG, lazy, operand, dims, operand->get_lazy()->get_dtype()->max()) {
            grad_enabled = false;
        }
        void enable_grad(bool enabled) override { grad_enabled = false; }
        const std::string &get_opname() const override { return opname; }
    };

    OpPtr empty_like(OpPtr op, DtypePtr dtype, DevicePtr device);
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
    OpPtr unsqueeze(OpPtr in_op, const ShapeDims &dims);
    OpPtr squeeze(OpPtr in_op, const ShapeDims &dims);
    OpPtr add(OpPtr lop, OpPtr rop);
    OpPtr sub(OpPtr lop, OpPtr rop);
    OpPtr mul(OpPtr lop, OpPtr rop);
    OpPtr div(OpPtr lop, OpPtr rop);
    OpPtr matmul(OpPtr lop, OpPtr rop);
    OpPtr inplace_add(OpPtr lop, OpPtr rop);
    OpPtr inplace_sub(OpPtr lop, OpPtr rop);
    OpPtr inplace_mul(OpPtr lop, OpPtr rop);
    OpPtr inplace_div(OpPtr lop, OpPtr rop);
    OpPtr eq(OpPtr lop, OpPtr rop);
    OpPtr neq(OpPtr lop, OpPtr rop);
    OpPtr lt(OpPtr lop, OpPtr rop);
    OpPtr gt(OpPtr lop, OpPtr rop);
    OpPtr leq(OpPtr lop, OpPtr rop);
    OpPtr geq(OpPtr lop, OpPtr rop);
    OpPtr minimum(OpPtr lop, OpPtr rop);
    OpPtr maximum(OpPtr lop, OpPtr rop);
    OpPtr sq(OpPtr in_op, bool in_place = false);
    OpPtr sqrt(OpPtr in_op, bool in_place = false);
    OpPtr neg(OpPtr in_op, bool in_place = false);
    OpPtr copy(OpPtr in_op);
    OpPtr exp(OpPtr in_op, bool in_place = false);
    OpPtr log(OpPtr in_op, bool in_place = false);
    OpPtr recip(OpPtr in_op, bool in_place = false);
    OpPtr reshape(OpPtr in_op, const ShapeView &view);
    OpPtr permute(OpPtr in_op, const ShapeDims &dims);
    OpPtr transpose(OpPtr in_op, isize start_dim, isize end_dim);
    OpPtr flatten(OpPtr in_op, isize start_dim, isize end_dim);
    OpPtr sum(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr mean(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr max(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr min(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr argmax(OpPtr in_op, const ShapeDims &dims = {});
    OpPtr argmin(OpPtr in_op, const ShapeDims &dims = {});

    template <typename T>
    concept Numeric = std::is_arithmetic_v<T>;

    template <typename T>
    concept NumericOrBool = std::is_arithmetic_v<T> || std::is_same_v<T, bool>;

    isize item(OpPtr op);

    template <Numeric T>
    OpPtr full(const ShapeView &view, T c, DtypePtr dtype, DevicePtr device) {
        LazyPtr lazy = Lazy::empty(Shape(view), dtype, device);
        OpPtr op = std::make_shared<FullOp>(lazy, view, dtype_cast_down(c, dtype), dtype);
        return op;
    }

    template <class T>
    OpPtr full_like(OpPtr in_op, T c, DtypePtr dtype, DevicePtr device) {
        return full(in_op->get_lazy()->get_view(), c, dtype, device);
    }

    template <Numeric T>
    OpPtr binary_with_scalar(OpPtr lop, T c, OpPtr (*op_func)(OpPtr, OpPtr)) {
        LazyPtr llazy = lop->get_lazy();
        DtypePtr ldtype = llazy->get_dtype();
        OpPtr rop = full(llazy->get_view(), c, ldtype, llazy->get_device());
        rop->enable_grad(false);
        return op_func(lop, rop);
    }

    template <NumericOrBool T>
    OpPtr eq_with_scalar(OpPtr lop, T c, OpPtr (*op_func)(OpPtr, OpPtr)) {
        LazyPtr llazy = lop->get_lazy();
        DtypePtr ldtype = llazy->get_dtype();
        OpPtr rop = full(llazy->get_view(), c, ldtype, llazy->get_device());
        rop->enable_grad(false);
        return op_func(lop, rop);
    }

    template <Numeric T>
    OpPtr add(OpPtr lop, T c) { return binary_with_scalar(lop, c, add); }

    template <Numeric T>
    OpPtr inplace_add(OpPtr lop, T c) { return binary_with_scalar(lop, c, inplace_add); }

    template <Numeric T>
    OpPtr sub(OpPtr lop, T c) { return binary_with_scalar(lop, c, sub); }

    template <Numeric T>
    OpPtr inplace_sub(OpPtr lop, T c) { return binary_with_scalar(lop, c, inplace_sub); }

    template <Numeric T>
    OpPtr mul(OpPtr lop, T c) { return binary_with_scalar(lop, c, mul); }

    template <Numeric T>
    OpPtr inplace_mul(OpPtr lop, T c) { return binary_with_scalar(lop, c, inplace_mul); }

    template <Numeric T>
    OpPtr div(OpPtr lop, T c) { return binary_with_scalar(lop, c, div); }

    template <Numeric T>
    OpPtr inplace_div(OpPtr lop, T c) { return binary_with_scalar(lop, c, inplace_div); }

    template <NumericOrBool T>
    OpPtr eq(OpPtr lop, T c) { return eq_with_scalar(lop, c, eq); }

    template <NumericOrBool T>
    OpPtr neq(OpPtr lop, T c) { return eq_with_scalar(lop, c, neq); }

    template <Numeric T>
    OpPtr lt(OpPtr lop, T c) { return binary_with_scalar(lop, c, lt); }

    template <Numeric T>
    OpPtr gt(OpPtr lop, T c) { return binary_with_scalar(lop, c, gt); }

    template <Numeric T>
    OpPtr leq(OpPtr lop, T c) { return binary_with_scalar(lop, c, leq); }

    template <Numeric T>
    OpPtr geq(OpPtr lop, T c) { return binary_with_scalar(lop, c, geq); }

    template <Numeric T>
    OpPtr minimum(OpPtr lop, T c) { return binary_with_scalar(lop, c, minimum); }

    template <Numeric T>
    OpPtr maximum(OpPtr lop, T c) { return binary_with_scalar(lop, c, maximum); }

    template <class O>
    OpPtr elmwise_binary(OpPtr lop, OpPtr rop) {
        LazyPtr llazy = lop->get_lazy();
        LazyPtr rlazy = rop->get_lazy();
        const ShapeView &lview = llazy->get_view();
        const ShapeView &rview = rlazy->get_view();
        DtypePtr ldtype = llazy->get_dtype();
        DtypePtr rdtype = rlazy->get_dtype();
        DevicePtr ldevice = llazy->get_device();
        DevicePtr rdevice = rlazy->get_device();

        if (!llazy->get_shape().broadcastable(rview)) {
            throw IncompatShapesForOp(O::opname, vnumstr(lview), vnumstr(rview));
        }
        if (!binary_dtypes.contains(ldtype) || ldtype != rdtype) {
            throw IncompatDtypesForOp(O::opname, ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice) {
            throw IncompatDevicesForOp(O::opname, ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_lop = broadcast(lop, rview);
        OpPtr broadcasted_rop = broadcast(rop, lview);
        LazyPtr out_lazy = Lazy::empty(Shape(broadcasted_lop->get_lazy()->get_view()), ldtype, ldevice);
        OpPtr out_op = std::make_shared<O>(out_lazy, broadcasted_lop, broadcasted_rop, false);
        return out_op;
    }

    template <class O>
    OpPtr inplace_binary(OpPtr lop, OpPtr rop) {
        LazyPtr llazy = lop->get_lazy();
        LazyPtr rlazy = rop->get_lazy();
        const Shape &lshape = llazy->get_shape();
        const ShapeView &lview = llazy->get_view();
        const ShapeView &rview = rlazy->get_view();
        DtypePtr ldtype = llazy->get_dtype();
        DtypePtr rdtype = rlazy->get_dtype();
        DevicePtr ldevice = llazy->get_device();
        DevicePtr rdevice = rlazy->get_device();

        if (!llazy->get_shape().broadcastable(rview)) {
            throw IncompatShapesForOp(O::opname, vnumstr(lview), vnumstr(rview));
        }
        if (!binary_dtypes.contains(ldtype) || ldtype != rdtype) {
            throw IncompatDtypesForOp(O::opname, ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice) {
            throw IncompatDevicesForOp(O::opname, ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_rop = broadcast_to(rop, lview);
        LazyPtr out_lazy = Lazy::empty(lshape, ldtype, ldevice);
        OpPtr out_op = std::make_shared<O>(out_lazy, lop, broadcasted_rop, true);
        return out_op;
    }

    template <class O>
    OpPtr unary(OpPtr in_op, bool in_place) {
        LazyPtr in_lazy = in_op->get_lazy();
        DtypePtr in_dtype = in_lazy->get_dtype();

        if (!float_dtype_by_dtype.contains(in_dtype)) {
            throw IncompatDtypeForOp(O::opname, in_dtype->str());
        }

        LazyPtr out_lazy = Lazy::empty(Shape(in_lazy->get_view()), in_dtype, in_lazy->get_device());
        OpPtr out_op = std::make_shared<O>(out_lazy, in_op, in_place);
        return out_op;
    }

    template <class O>
    OpPtr unary_float(OpPtr in_op, bool in_place) {
        LazyPtr in_lazy = in_op->get_lazy();
        DtypePtr in_dtype = in_lazy->get_dtype();

        if (in_place) {
            if (in_dtype->get_type() != DtypeType::FLOAT) {
                // This method requires the operand to be of floating-point type
                // to do in-place operation since the result is of floating-point type
                throw IncompatDtypeForOp(O::opname, in_dtype->str());
            }
        }

        auto result_dtype = float_dtype_by_dtype.find(in_dtype);
        if (result_dtype == float_dtype_by_dtype.end()) {
            throw IncompatDtypeForOp(O::opname, in_dtype->str());
        }

        LazyPtr out_lazy = Lazy::empty(Shape(in_lazy->get_view()), result_dtype->second, in_lazy->get_device());
        OpPtr out_op = std::make_shared<O>(out_lazy, in_op, in_place);
        return out_op;
    }

    template <class O>
    OpPtr cmp(OpPtr lop, OpPtr rop, DtypePtrSet &valid_dtypes) {
        LazyPtr llazy = lop->get_lazy();
        LazyPtr rlazy = rop->get_lazy();
        const ShapeView &lview = llazy->get_view();
        const ShapeView &rview = rlazy->get_view();
        DtypePtr ldtype = llazy->get_dtype();
        DtypePtr rdtype = rlazy->get_dtype();
        DevicePtr ldevice = llazy->get_device();
        DevicePtr rdevice = rlazy->get_device();

        if (!llazy->get_shape().broadcastable(rview)) {
            throw IncompatShapesForOp(O::opname, vnumstr(lview), vnumstr(rview));
        }
        if (!valid_dtypes.contains(ldtype) || ldtype != rdtype) {
            throw IncompatDtypesForOp(O::opname, ldtype->str(), rdtype->str());
        }
        if (ldevice != rdevice) {
            throw IncompatDevicesForOp(O::opname, ldevice->str(), rdevice->str());
        }

        OpPtr broadcasted_lop = broadcast(lop, rview);
        OpPtr broadcasted_rop = broadcast(rop, lview);
        LazyPtr out_lazy = Lazy::empty(Shape(broadcasted_lop->get_lazy()->get_view()), &b8, ldevice);
        OpPtr out_op = std::make_shared<O>(out_lazy, broadcasted_lop, broadcasted_rop);
        return out_op;
    }

    template <class O>
    OpPtr reduce(OpPtr in_op, const ShapeDims &dims, DtypePtr result_dtype, DtypePtrSet &valid_dtypes) {
        LazyPtr in_lazy = in_op->get_lazy();
        const Shape &in_shape = in_lazy->get_shape();
        DtypePtr in_dtype = in_lazy->get_dtype();
        DevicePtr in_device = in_lazy->get_device();

        if (!valid_dtypes.contains(in_dtype)) {
            throw IncompatDtypeForOp(O::opname, in_dtype->str());
        }

        LazyPtr reduction_arr;
        OpPtr reduction_op;

        if (dims.size() == 0) {
            // Reduce to one element
            reduction_arr = Lazy::empty(Shape({1}), result_dtype, in_device);
            reduction_op = std::make_shared<O>(reduction_arr, in_op, dims);
            return reduction_op;
        }

        // Fill kept_dims from 0 to ndim-1
        ShapeDims kept_dims(in_shape.get_ndim());
        ShapeDims reduction_dims;
        std::iota(kept_dims.begin(), kept_dims.end(), 0);

        // Remove the dimensions to be reduced from kept_dims and add them to reduction_dims
        for (auto &dim : dims) {
            auto iter = std::find(kept_dims.begin(), kept_dims.end(), dim);
            if (iter == kept_dims.end()) {
                throw std::invalid_argument("Invalid reduction dimension " + std::to_string(dim) + " on array " + in_lazy->get_id().str() + ".");
            } else {
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

        for (auto &dim : kept_dims) {
            kept_view.emplace_back(in_shape[dim]);
            kept_numel *= in_shape[dim];
        }
        for (auto &dim : reduction_dims) {
            reduction_numel *= in_shape[dim];
        }

        OpPtr reshape_op_before_reduction = reshape(permutation_op, {kept_numel, reduction_numel});
        // Reduce the array
        reduction_arr = Lazy::empty(Shape({kept_numel, 1}), result_dtype, in_device);
        reduction_op = std::make_shared<O>(reduction_arr, reshape_op_before_reduction, dims);
        // Reshape the array back to the shape without reduced dimensions(except for 1 at the end)
        kept_view.emplace_back(1);
        OpPtr reshape_op_after_reduction = reshape(reduction_op, kept_view);
        return reshape_op_after_reduction;
    }
} // namespace ax::graph