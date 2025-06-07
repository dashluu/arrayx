#pragma once

#include "../array/array.h"

namespace ax::nn {
    using namespace ax::array;

    struct JitKey {
    private:
        ArrayVec arrays;

    public:
        JitKey(const ArrayVec &arrays) : arrays(arrays) {}

        JitKey(const JitKey &other) : arrays(other.arrays) {}

        JitKey &operator=(const JitKey &other) {
            arrays = other.arrays;
            return *this;
        }

        bool operator==(const JitKey &other) const {
            if (arrays.size() != other.arrays.size()) {
                return false;
            }
            return std::all_of(arrays.begin(), arrays.end(), [&other, i = 0](const Array &arr) mutable {
                return arr.get_shape() == other.arrays[i].get_shape() &&
                       arr.get_dtype() == other.arrays[i].get_dtype() &&
                       arr.get_device() == other.arrays[i].get_device() &&
                       ++i;
            });
        }

        std::size_t hash() const {
            std::size_t seed = 0;
            // Manual hash_combine implementation
            auto hash_combine = [](std::size_t &seed, const auto &v) {
                seed ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };

            for (const Array &arr : arrays) {
                hash_combine(seed, std::hash<Shape>{}(arr.get_shape()));
                hash_combine(seed, arr.get_dtype());
                hash_combine(seed, arr.get_device());
            }

            return seed;
        }
    };
} // namespace ax::nn

namespace std {
    template <>
    struct hash<ax::nn::JitKey> {
        std::size_t operator()(const ax::nn::JitKey &key) const { return key.hash(); }
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
        Array operator()(const ArrayVec &input, F &&f) {
            JitKey key(input);
            if (cache.count(key)) {
                return cache[key];
            } else {
                Array result = f(input);
                result.compile();
                cache[key] = result;
                return result;
            }
        }
    };
} // namespace ax::nn