#pragma once

#include "../utils.h"

namespace ax::devices
{
    using ax::core::isize;
    using ax::core::IStr;

    enum struct DeviceType
    {
        CPU,
        MPS
    };

    struct Device : public IStr, public std::enable_shared_from_this<Device>
    {
    private:
        DeviceType type;
        isize id;
        std::string name;

        void init_name_from_type_id()
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
            name = typestr + ":" + std::to_string(id);
        }

    public:
        Device() = delete;

        Device(DeviceType type, isize id) : type(type), id(id) { init_name_from_type_id(); }

        Device(const Device &device) : Device(device.type, device.id) { init_name_from_type_id(); }

        DeviceType get_type() const { return type; }

        isize get_id() const { return id; }

        const std::string &get_name() const { return name; }

        Device &operator=(const Device &device) = delete;

        bool operator==(const Device &device) const { return type == device.type && id == device.id; }

        bool operator!=(const Device &device) const { return !(*this == device); }

        const std::string str() const override { return get_name(); }
    };

    using DevicePtr = std::shared_ptr<Device>;
}

namespace std
{
    template <>
    struct hash<const ax::devices::Device *>
    {
        std::size_t operator()(const ax::devices::Device *device) const
        {
            return std::hash<std::string>()(device->get_name());
        }
    };

    template <>
    struct equal_to<const ax::devices::Device *>
    {
        bool operator()(const ax::devices::Device *lhs, const ax::devices::Device *rhs) const
        {
            return *lhs == *rhs;
        }
    };
}