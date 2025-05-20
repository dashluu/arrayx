#pragma once

#include "../utils.h"

namespace ax::core
{
    struct Id : public IStr
    {
    private:
        isize data;

    public:
        Id() : data(0) {}
        Id(isize id) : data(id) {}
        Id(const Id &id) : data(id.data) {}
        bool operator==(const Id &id) const { return data == id.data; }
        bool operator!=(const Id &id) const { return !(*this == id); }
        Id &operator=(const Id &id)
        {
            data = id.data;
            return *this;
        }
        isize get_data() const { return data; }
        const std::string str() const override { return std::to_string(data); }
    };

    struct IdGenerator
    {
    private:
        static isize counter;

    public:
        IdGenerator() = default;
        IdGenerator(const IdGenerator &) = delete;
        IdGenerator &operator=(const IdGenerator &) = delete;

        Id generate()
        {
            Id curr(counter++);
            return curr;
        }
    };

    inline isize IdGenerator::counter = 1;
}

namespace std
{
    template <>
    struct hash<ax::core::Id>
    {
        std::size_t operator()(const ax::core::Id &id) const
        {
            return std::hash<ax::core::isize>()(id.get_data());
        }
    };
}