#pragma once

#include "../../allocator.h"

namespace ax::runtime::metal
{
	using ax::runtime::Allocator;

	struct MTLAllocator : public Allocator
	{
		uint8_t *alloc(isize nbytes) override
		{
			allocated += nbytes;
			auto ptr = new uint8_t[nbytes];
			std::memset(ptr, 0, nbytes);
			return ptr;
		}

		void free(uint8_t *ptr, isize nbytes) override
		{
			allocated -= nbytes;
			delete[] ptr;
		}
	};
}