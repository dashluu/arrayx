#pragma once

#include "../array/array.h"

namespace ax::nn
{
	using namespace ax::array;

	struct ModuleKey
	{
	private:
		Array arr;

	public:
		ModuleKey(const Array &arr) : arr(arr) {}

		ModuleKey(const ModuleKey &other) : arr(other.arr) {}

		ModuleKey &operator=(const ModuleKey &other)
		{
			arr = other.arr;
			return *this;
		}

		bool operator==(const ModuleKey &other) const
		{
			return arr.get_shape() == other.arr.get_shape() &&
				   arr.get_dtype() == other.arr.get_dtype() &&
				   arr.get_device() == other.arr.get_device();
		}

		bool operator!=(const ModuleKey &other) const { return !(*this == other); }

		const Array &get_array() const { return arr; }
	};
}

namespace std
{
	template <>
	struct hash<ax::nn::ModuleKey>
	{
		std::size_t operator()(const ax::nn::ModuleKey &key) const
		{
			std::size_t seed = 0;

			// Manual hash_combine implementation
			auto hash_combine = [](std::size_t &seed, const auto &v)
			{
				seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			hash_combine(seed, std::hash<ax::core::Shape>{}(key.get_array().get_shape()));
			hash_combine(seed, key.get_array().get_dtype());
			hash_combine(seed, key.get_array().get_device());
			return seed;
		}
	};
}

namespace ax::nn
{
	class Module
	{
	protected:
		Array input;
		static std::unordered_map<ModuleKey, Array> modules;

	public:
		Module() = default;

		virtual ~Module() = default;

		Module(const Module &) = delete;

		Module &operator=(const Module &) = delete;

		virtual Array forward(const Array &input) = 0;

		Array operator()(const Array &input)
		{
			const ModuleKey key(input);
			Array output;

			if (modules.find(key) == modules.end())
			{
				// Initialize a new compute primitive
				this->input = input.detach();
				output = forward(this->input);
				modules[key] = output;
			}
			else
			{
				// Update the input lazy array
				this->input.set_lazy(input);
				output = modules[key];
			}

			output.eval();
			return output;
		}
	};

	inline std::unordered_map<ModuleKey, Array> Module::modules;
}