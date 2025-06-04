#pragma once

#include "module.h"

namespace ax::nn
{
	ArrayPtr relu(ArrayPtr x)
	{
		return x->geq(0)->astype(&f32)->mul(x);
	}
}