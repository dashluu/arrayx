#include "array.h"

using namespace ax;

int main()
{
	Backend::init();
	auto x1 = Array::arange({2, 3, 4}, 0, 2, &f32, "mps:0");
	auto x2 = Array::arange({2, 3, 4}, 1, 4, &f32, "mps:0");
	auto x3 = x1->add(x2);
	x3->eval();
	std::cout << x3->str() << std::endl;
	Backend::shutdown();
	return 0;
}