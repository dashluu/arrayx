#pragma once

#include "../utils.h"

namespace ax::core
{
    enum struct DtypeName
    {
        F32,
        F64,
        I8,
        I16,
        I32,
        I64,
        B8
    };

    enum struct DtypeType
    {
        FLOAT,
        INT,
        BOOL
    };

    inline const std::unordered_map<DtypeName, const std::string> str_by_dtype_name = {
        {DtypeName::F32, "f32"},
        {DtypeName::I8, "i8"},
        {DtypeName::I16, "i16"},
        {DtypeName::I32, "i32"},
        {DtypeName::B8, "b8"}};

    struct Dtype
    {
    private:
        DtypeName name;
        DtypeType type;
        isize size;

    public:
        Dtype(DtypeName name, DtypeType type, isize size) : name(name), type(type), size(size) {}

        Dtype(const Dtype &dtype) : Dtype(dtype.name, dtype.type, dtype.size) {}

        virtual ~Dtype() = default;

        DtypeName get_name() const { return name; }

        const std::string &get_name_str() const { return str_by_dtype_name.at(name); }

        DtypeType get_type() const { return type; }

        isize get_size() const { return size; }

        bool operator==(const Dtype &dtype) const { return name == dtype.name; }

        bool operator!=(const Dtype &dtype) const { return !(*this == dtype); }

        Dtype &operator=(const Dtype &dtype)
        {
            name = dtype.name;
            type = dtype.type;
            size = dtype.size;
            return *this;
        }

        std::string str() const
        {
            return get_name_str();
        }

        bool is_numeric() const { return type != DtypeType::BOOL; }

        bool is_bool() const { return type == DtypeType::BOOL; }

        virtual std::string get_value_as_str(uint8_t *ptr) const = 0;

        virtual std::string get_value_as_str(isize val) const = 0;

        virtual isize max() const = 0;

        virtual isize min() const = 0;
    };

    template <class T>
    struct Float : public Dtype
    {
    public:
        Float(DtypeName name, isize size) : Dtype(name, DtypeType::FLOAT, size) {}

        Float(const Float<T> &dtype) : Dtype(dtype) {}

        std::string get_value_as_str(uint8_t *ptr) const override
        {
            T val = *reinterpret_cast<T *>(ptr);
            if (0 < val && val <= 1e-5)
            {
                return std::format("{:.4e}", val);
            }
            return std::format("{:.4f}", val);
        }
    };

    template <class T>
    struct Int : public Dtype
    {
    public:
        Int(DtypeName name, isize size) : Dtype(name, DtypeType::INT, size) {}

        Int(const Int<T> &dtype) : Dtype(dtype) {}

        std::string get_value_as_str(uint8_t *ptr) const override
        {
            return std::to_string(*reinterpret_cast<T *>(ptr));
        }

        std::string get_value_as_str(isize val) const override
        {
            return std::to_string(val);
        }

        isize max() const override { return std::numeric_limits<T>::max(); }

        isize min() const override { return std::numeric_limits<T>::lowest(); }
    };

    struct F32 : public Float<float>
    {
    public:
        F32() : Float<float>(DtypeName::F32, 4) {}
        F32(const F32 &dtype) : Float<float>(dtype) {}

        std::string get_value_as_str(isize val) const override
        {
            return std::to_string(std::bit_cast<float>(static_cast<int>(val)));
        }

        isize max() const override { return std::bit_cast<int>(std::numeric_limits<float>::max()); }

        isize min() const override { return std::bit_cast<int>(std::numeric_limits<float>::lowest()); }
    };

    struct F64 : public Float<double>
    {
    public:
        F64() : Float<double>(DtypeName::F64, 8) {}
        F64(const F64 &dtype) : Float<double>(dtype) {}

        std::string get_value_as_str(isize val) const override
        {
            return std::to_string(std::bit_cast<double>(static_cast<int64_t>(val)));
        }

        isize max() const override { return std::bit_cast<int64_t>(std::numeric_limits<double>::max()); }

        isize min() const override { return std::bit_cast<int64_t>(std::numeric_limits<double>::lowest()); }
    };

    struct I8 : public Int<int8_t>
    {
    public:
        I8() : Int<int8_t>(DtypeName::I8, 1) {}
        I8(const I8 &dtype) : Int<int8_t>(dtype) {}
    };

    struct I16 : public Int<int16_t>
    {
    public:
        I16() : Int<int16_t>(DtypeName::I16, 2) {}
        I16(const I16 &dtype) : Int<int16_t>(dtype) {}
    };

    struct I32 : public Int<int32_t>
    {
    public:
        I32() : Int<int32_t>(DtypeName::I32, 4) {}
        I32(const I32 &dtype) : Int<int32_t>(dtype) {}
    };

    struct I64 : public Int<int64_t>
    {
    public:
        I64() : Int<int64_t>(DtypeName::I64, 8) {}
        I64(const I64 &dtype) : Int<int64_t>(dtype) {}
    };

    struct Bool : public Dtype
    {
    public:
        Bool() : Dtype(DtypeName::B8, DtypeType::BOOL, 1) {}

        Bool(const Bool &dtype) : Dtype(dtype) {}

        std::string get_value_as_str(uint8_t *ptr) const override
        {
            return *ptr ? "True" : "False";
        }

        std::string get_value_as_str(isize val) const override
        {
            return std::to_string(static_cast<bool>(val));
        }

        isize max() const override { return std::numeric_limits<bool>::max(); }

        isize min() const override { return std::numeric_limits<bool>::lowest(); }
    };

    inline const F32 f32;
    inline const F64 f64;
    inline const I8 i8;
    inline const I16 i16;
    inline const I32 i32;
    inline const I64 i64;
    inline const Bool b8;

    using DtypePtr = const Dtype *;
    using DtypePtrSet = const std::unordered_set<DtypePtr>;
}

namespace std
{
    template <>
    struct hash<const ax::core::Dtype *>
    {
        std::size_t operator()(const ax::core::Dtype *dtype) const
        {
            return std::hash<std::string>()(dtype->get_name_str());
        }
    };

    template <>
    struct equal_to<const ax::core::Dtype *>
    {
        bool operator()(const ax::core::Dtype *lhs, const ax::core::Dtype *rhs) const
        {
            return *lhs == *rhs;
        }
    };
}

namespace ax::core
{
    inline DtypePtrSet all_dtypes = {&b8, &i32, &f32};
    inline DtypePtrSet numeric_dtypes = {&i32, &f32};
    inline DtypePtrSet binary_dtypes = {&i32, &f32};
    inline DtypePtrSet unary_dtypes = {&i32, &f32};
    inline DtypePtrSet cmp_dtypes = {&i32, &f32};
    inline DtypePtrSet eq_dtypes = {&i32, &f32, &b8};
    inline const std::unordered_map<DtypePtr, DtypePtr> float_dtype_by_dtype = {
        {&i32, &f32},
        {&f32, &f32}};

    template <class T>
    isize dtype_cast(T c, DtypePtr dtype)
    {
        switch (dtype->get_type())
        {
        case DtypeType::FLOAT:
            switch (dtype->get_size())
            {
            default:
                return std::bit_cast<int>(static_cast<float>(c));
            }
        case DtypeType::INT:
            return static_cast<isize>(c);
        default:
            return static_cast<bool>(c);
        }
    }
}