#pragma once

#include "../device/device.h"
#include "../runtime/runner.h"

namespace ax::array {
    using namespace ax::device;
    using namespace ax::runtime;
    using GraphBuilder = std::function<std::shared_ptr<ComputeGraph>(OpPtr)>;

    class Backend {
    private:
        std::unordered_map<std::string, DevicePtr> devices;
        std::unordered_map<std::string, RunnerPtr> runners;
        std::unordered_map<std::string, GraphBuilder> graph_builders;

    public:
        Backend() = default;
        Backend(const Backend &) = delete;
        Backend &operator=(const Backend &) = delete;
        DevicePtr get_device(const std::string &name) const;
        RunnerPtr get_runner(const std::string &device_name) const;
        GraphBuilder get_graph_builder(const std::string &device_name) const;
        size_t count_devices() const { return devices.size(); }
        void init();
    };
} // namespace ax::array