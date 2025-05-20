#include "iter.h"

namespace ax::core
{
	uint8_t *Array::strided_elm_ptr(isize k) const
	{
		if (is_contiguous())
		{
			return get_ptr() + k * get_itemsize();
		}
		std::vector<isize> idx(get_ndim());
		isize carry = k;
		isize tmp;
		for (int i = get_ndim() - 1; i >= 0; i--)
		{
			tmp = carry;
			idx[i] = tmp % shape[i];
			carry = tmp / shape[i];
		}
		const ShapeStride &stride = get_stride();
		uint8_t *ptr = get_ptr();
		for (size_t i = 0; i < idx.size(); i++)
		{
			ptr += idx[i] * stride[i] * get_itemsize();
		}
		return ptr;
	}

	const std::string Array::str() const
	{
		auto iter = std::make_unique<ArrayIter>(shared_from_this());
		iter->start();
		bool next_elm_available = iter->has_next();
		std::string s = "";
		for (int i = 0; i < get_ndim(); i++)
		{
			s += "[";
		}
		if (!next_elm_available)
		{
			for (int i = 0; i < get_ndim(); i++)
			{
				s += "]";
			}
			return s;
		}
		ShapeView elms_per_dim = shape.get_elms_per_dim();
		int close = 0;
		while (next_elm_available)
		{
			close = 0;
			uint8_t *ptr = iter->next();
			// std::cout << std::hex << static_cast<void *>(ptr) << std::endl;
			s += dtype->get_value_as_str(ptr);
			for (int i = elms_per_dim.size() - 1; i >= 0; i--)
			{
				if (iter->count() % elms_per_dim[i] == 0)
				{
					s += "]";
					close += 1;
				}
			}
			next_elm_available = iter->has_next();
			if (next_elm_available)
			{
				if (close > 0)
				{
					s += ", \n";
				}
				else
				{
					s += ", ";
				}
				for (int i = 0; i < close; i++)
				{
					s += "[";
				}
			}
		}
		return s;
	}
}