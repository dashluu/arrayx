#pragma once

#include "../utils.h"
#include "allocator.h"

namespace ax::core
{
    enum class DeviceType
    {
        CPU,
        MPS
    };

    struct Device : public IStr
    {
    private:
        Allocator *allocator;
        DeviceType type;
        isize idx;

    public:
        Device() = delete;

        Device(DeviceType type, Allocator *allocator, isize idx = 0) : type(type), idx(idx), allocator(allocator) {}

        Device(const Device &device) : Device(device.type, device.allocator, device.idx) {}

        DeviceType get_type() const { return type; }

        isize get_idx() const { return idx; }

        Allocator *get_allocator() const { return allocator; }

        Device &operator=(const Device &device) = delete;

        bool operator==(const Device &device) const { return type == device.type && idx == device.idx; }

        bool operator!=(const Device &device) const { return !(*this == device); }

        const std::string str() const override
        {
            std::string typestr;
            switch (type)
            {
            case DeviceType::CPU:
                typestr = "cpu";
                break;
            default:
                typestr = "mps";
                break;
            }
            return typestr + ":" + std::to_string(idx);
        }
    };

    inline const Device device0(DeviceType::MPS, &allocator0, 0);
    using DevicePtr = const Device *;
}