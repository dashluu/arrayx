#pragma once

#include "../utils.h"

namespace ax::runtime
{
	class RunnerContext : public std::enable_shared_from_this<RunnerContext>
	{
	public:
		RunnerContext() = default;
		RunnerContext(const RunnerContext &) = delete;
		virtual ~RunnerContext() = default;
		RunnerContext &operator=(const RunnerContext &) = delete;
	};

	using RunnerContextPtr = std::shared_ptr<RunnerContext>;
}