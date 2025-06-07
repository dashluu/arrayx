#pragma once

#include "../array/array.h"

namespace ax::nn {
    using namespace ax::array;

    struct JitKey {
    private:
        Array arr;

    public:
        JitKey(const Array &arr) : arr(arr) {}

        JitKey(const JitKey &other) : arr(other.arr) {}

        JitKey &operator=(const JitKey &other) {
            arr = other.arr;
            return *this;
        }

        bool operator==(const JitKey &other) const {
            return arr.get_shape() == other.arr.get_shape() &&
                   arr.get_dtype() == other.arr.get_dtype() &&
                   arr.get_device() == other.arr.get_device();
        }

        const Array &get_array() const { return arr; }
    };
} // namespace ax::nn

namespace std {
    template <>
    struct hash<ax::nn::JitKey> {
        std::size_t operator()(const ax::nn::JitKey &key) const {
            std::size_t seed = 0;

            // Manual hash_combine implementation
            auto hash_combine = [](std::size_t &seed, const auto &v) {
                seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };

            hash_combine(seed, std::hash<ax::core::Shape>{}(key.get_array().get_shape()));
            hash_combine(seed, key.get_array().get_dtype());
            hash_combine(seed, key.get_array().get_device());
            return seed;
        }
    };
} // namespace std

namespace ax::nn {
    class Jit {

    private:
        std::unordered_map<JitKey, Array> cache;

    public:
        Jit() = default;
        Jit(const Jit &) = delete;
        Jit &operator=(const Jit &) = delete;

        template <class F>
        Array operator()(const Array &input, F &&f) {
            JitKey key(input);
            if (cache.count(key)) {
                std::cout << "Loading JIT cache for input array " << key.get_array().get_id().str() << "..." << std::endl;
                return cache[key];
            } else {
                std::cout << "JIT-caching input array " << key.get_array().get_id().str() << "..." << std::endl;
                Array result = f(input);
                result.compile();
                cache[key] = result;
                return result;
            }
        }
    };
} // namespace ax::nn