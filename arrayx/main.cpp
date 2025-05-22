#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "device/device_manager.h"
#include "graph/compute_graph.h"
#include "runtime/metal/runner/mtl_runner.h"

using namespace ax::core;
using namespace ax::runtime::metal;
using namespace ax::graph;

int main()
{
	DevicePtr device = DeviceManager::get_instance().get_device("mps");
	auto op1 = arange({2, 3, 4}, 0, 2, &f32, device);
	// auto op2 = arange({2, 3, 4}, 1, 4, &f32, device);
	// auto op3 = add(op1, 2);
	auto op3 = sum(op1);
	auto cg = std::make_shared<ComputeGraph>(op3);
	std::shared_ptr<DeviceContext> ctx = DeviceManager::get_instance().get_context("mps");
	MTLRunner runner(std::static_pointer_cast<MTLContext>(ctx));
	cg->forward();
	runner.forward(cg);
	auto x = op3->get_array();
	std::cout << x->str() << std::endl;
	return 0;
}