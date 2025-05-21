#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "graph/compute_graph.h"
#include "runtime/metal/runner/mtl_runner.h"

using namespace ax::core;
using namespace ax::runtime::metal;
using namespace ax::graph;

int main()
{
	auto op1 = arange({2, 3, 4}, 0, 2);
	// auto op2 = arange({2, 3, 4}, 1, 4);
	// auto op3 = add(op1, 2);
	auto op3 = sum(op1);
	auto cg = std::make_shared<ComputeGraph>(op3);
	auto ctx = std::make_shared<MTLContext>("runtime/metal/kernels/kernels.metallib");
	MTLRunner runner(ctx);
	cg->forward();
	runner.forward(cg);
	auto x = op3->get_output();
	std::cout << x->str() << std::endl;
	return 0;
}