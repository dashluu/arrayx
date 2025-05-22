#pragma once

#include "../allocator.h"

namespace ax::devices::metal
{
	using ax::devices::Allocator;

	struct MTLAllocator : public Allocator
	{
	public:
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