#include "array/array.h"

using namespace ax::array;

int main()
{
	Backend::init();
	auto x1 = Array::arange({2, 3, 4}, 0, 2, &f32, "mps:0");
	// auto x2 = Array::arange({2, 3, 4}, 1, 4, &f32, "mps:0");
	auto x3 = x1->sum();
	x3->eval();
	std::cout << x3->str() << std::endl;
	Backend::cleanup();
	return 0;
}