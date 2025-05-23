#include "backend.h"

#ifdef __APPLE__
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "runtime/metal/mtl_runner.h"
#endif

namespace ax
{
	static Backend backend;

	void Backend::init()
	{
		if (backend.devices.size() > 0)
		{
			return;
		}

		// TODO: assume there is a CPU for now
		auto cpu = std::make_shared<Device>(DeviceType::CPU, 0);
		backend.devices.emplace("cpu", cpu);
		backend.devices.emplace(cpu->get_name(), cpu);

#ifdef __APPLE__
		NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
		// Get all available Metal devices
		NS::Array *mtl_devices = MTL::CopyAllDevices();

		if (!mtl_devices)
		{
			return;
		}

		DevicePtr device;
		RunnerPtr runner;
		std::shared_ptr<ax::runtime::metal::MTLContext> context;
		const std::string &lib_path = "runtime/metal/kernels/kernels.metallib";

		for (NS::UInteger i = 0; i < mtl_devices->count(); ++i)
		{
			MTL::Device *mtl_device = mtl_devices->object<MTL::Device>(i);
			device = std::make_shared<Device>(DeviceType::MPS, i);
			context = std::make_shared<ax::runtime::metal::MTLContext>(mtl_device, lib_path);
			runner = std::make_shared<ax::runtime::metal::MTLRunner>(context);
			backend.devices.emplace(device->get_name(), device);
			backend.runners.emplace(device->get_name(), runner);
			std::cout << "Initializing device " << device->get_name() << "..." << std::endl;
		}

		pool->release();
#endif
	}

	void Backend::shutdown()
	{
		backend.devices.clear();
		backend.runners.clear();
	}

	const Backend &Backend::instance()
	{
		return backend;
	}

	DevicePtr Backend::get_device(const std::string &name) const
	{
		return devices.find(name) == devices.end() ? nullptr : devices.at(name);
	}

	RunnerPtr Backend::get_runner(const std::string &device_name) const
	{
		return runners.find(device_name) == runners.end() ? nullptr : runners.at(device_name);
	}
}