#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ax::core {
    class LazyArray;
    using LazyArrayPtr = std::shared_ptr<LazyArray>;
    using isize = int64_t;

    template <class T>
    inline uint64_t vsize(const std::vector<T> &v) {
        return v.size() * sizeof(T);
    }

    template <class T>
    inline const std::string vstr(const std::vector<T> &v, const std::function<std::string(T)> &f) {
        std::string s = "";
        for (size_t i = 0; i < v.size(); i++) {
            s += f(v[i]);
            if (i < v.size() - 1) {
                s += ", ";
            }
        }
        return s;
    }

    template <class T>
    inline const std::string vnumstr(const std::vector<T> &v) {
        return vstr<T>(v, [](T a) { return std::to_string(a); });
    }
} // namespace ax::core