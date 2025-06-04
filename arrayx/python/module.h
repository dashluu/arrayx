#pragma once

#include "bind.h"

namespace ax::bind
{
	struct PyModule : axnn::Module
	{
		NB_TRAMPOLINE(axnn::Module, 1);

		axr::Array forward(const axr::Array &input) override
		{
			NB_OVERRIDE_PURE(forward, input);
		}
	};
}