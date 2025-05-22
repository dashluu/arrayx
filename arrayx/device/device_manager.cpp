#include "device_manager.h"

#ifdef __APPLE__
#include "metal/mtl_context.h"
#endif

namespace ax::device
{
	void DeviceManager::init()
	{
		// TODO: assume there is a CPU for now
		auto cpu = std::make_shared<Device>(DeviceType::CPU, 0);
		devices.insert({"cpu", cpu});
		devices.insert({cpu->get_name(), cpu});

#ifdef __APPLE__
		NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
		// Get all available Metal devices
		NS::Array *mtl_devices = MTL::CopyAllDevices();

		if (!mtl_devices)
		{
			return;
		}

		DevicePtr device;
		DeviceContextPtr context;
		const std::string &lib_path = "runtime/metal/kernels/kernels.metallib";

		for (NS::UInteger i = 0; i < mtl_devices->count(); ++i)
		{
			MTL::Device *mtl_device = mtl_devices->object<MTL::Device>(i);
			device = std::make_shared<Device>(DeviceType::MPS, i);
			context = std::make_shared<ax::device::metal::MTLContext>(mtl_device, lib_path);

			if (i == 0)
			{
				// Default MPS device
				devices.insert({"mps", device});
				contexts.insert({"mps", context});
			}

			devices.insert({device->get_name(), device});
			contexts.insert({device->get_name(), context});
		}

		pool->release();
#endif
	}

	const DeviceManager &DeviceManager::get_instance()
	{
		static DeviceManager instance;
		if (instance.devices.empty())
		{
			instance.init();
		}
		return instance;
	}

	DevicePtr DeviceManager::get_device(const std::string &name) const
	{
		return devices.find(name) == devices.end() ? nullptr : devices.at(name);
	}

	DeviceContextPtr DeviceManager::get_context(const std::string &device_name) const
	{
		return contexts.find(device_name) == contexts.end() ? nullptr : contexts.at(device_name);
	}
}