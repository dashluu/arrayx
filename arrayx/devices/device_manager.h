#pragma once

#include "device.h"
#include "device_context.h"

namespace ax::devices
{
	class DeviceManager
	{
	private:
		std::unordered_map<std::string, DevicePtr> devices;
		std::unordered_map<std::string, DeviceContextPtr> contexts;
		void init();
		DeviceManager() = default;

	public:
		static const DeviceManager &get_instance();
		DeviceManager(const DeviceManager &) = delete;
		DeviceManager &operator=(const DeviceManager &) = delete;
		DevicePtr get_device(const std::string &name);
		DeviceContextPtr get_context(const std::string &device_name);
	};
}