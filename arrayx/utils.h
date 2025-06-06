#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <ranges>
#include <memory>
#include <iomanip>
#include <numeric>
#include <cstdlib>
#include <cmath>
#include <list>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <bit>

namespace ax::core
{
    class LazyArray;
    using LazyArrayPtr = std::shared_ptr<LazyArray>;
    using isize = int64_t;

    template <class T>
    inline uint64_t vsize(const std::vector<T> &v)
    {
        return v.size() * sizeof(T);
    }

    template <class T>
    inline const std::string vstr(const std::vector<T> &v, const std::function<std::string(T)> &f)
    {
        std::string s = "";
        for (size_t i = 0; i < v.size(); i++)
        {
            s += f(v[i]);
            if (i < v.size() - 1)
            {
                s += ", ";
            }
        }
        return s;
    }

    template <class T>
    inline const std::string vnumstr(const std::vector<T> &v)
    {
        return vstr<T>(v, [](T a)
                       { return std::to_string(a); });
    }
}