#include "array/array.h"

using namespace ax::array;

int main() {
    Backend::init();
    auto x1 = Array::ones({2, 3, 4});
    auto x2 = Array::ones({1, 3, 1});
    auto x3 = x1 + x2;
    std::cout << x3.str() << std::endl;
    Backend::cleanup();
    return 0;
}