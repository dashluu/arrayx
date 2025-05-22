#pragma once

#include "../utils.h"

namespace ax::devices
{
	class DeviceContext : public std::enable_shared_from_this<DeviceContext>
	{
	public:
		DeviceContext() = default;
		DeviceContext(const DeviceContext &) = delete;
		virtual ~DeviceContext() = default;
		DeviceContext &operator=(const DeviceContext &) = delete;
	};

	using DeviceContextPtr = std::shared_ptr<DeviceContext>;
}