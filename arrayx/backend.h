#pragma once

#include "device/device.h"
#include "runtime/runner.h"

namespace ax
{
	using namespace ax::device;
	using namespace ax::runtime;

	class Backend
	{
	private:
		std::unordered_map<std::string, DevicePtr> devices;
		std::unordered_map<std::string, RunnerPtr> runners;

	public:
		Backend() = default;
		Backend(const Backend &) = delete;
		Backend &operator=(const Backend &) = delete;
		DevicePtr get_device(const std::string &name) const;
		RunnerPtr get_runner(const std::string &device_name) const;
		size_t count_devices() const { return devices.size(); }
		static void init();
		static void shutdown();
		static const Backend &instance();
	};
}