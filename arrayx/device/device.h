#pragma once

#include "../utils.h"

namespace ax::device
{
    using ax::core::isize;

    enum struct DeviceType
    {
        CPU,
        MPS
    };

    struct Device : public std::enable_shared_from_this<Device>
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

        const std::string str() const { return get_name(); }
    };

    using DevicePtr = std::shared_ptr<Device>;

    const std::string default_device_name = "mps:0";
}