#include <torch/extension.h>

torch::Tensor trilinear_fw_cu( //fw means forward
	torch::Tensor feats,
	torch::Tensor points	
){
	const int N = feats.size(0), F = feats.size(2);
	torch::Tensor feat_interp = torch::zeros({N, F}, feats.options()); // cpp form
	// python form:
	// feat_interp = torch.zeros(N, F, dtype=torch.float32, device='cuda:1')
	// what if we want to define a variable which have different dtype with input, 
	//such as integer, with cpp:
	// torch::Tensor feat_interp = torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device()))
	const dim3 threads(16, 16); // define size of thread
	const dim3 blocks((N+threads.x-1)/threads.x,
			(F+threads.y-1)/threads.y); //define size of block
	// lunach kernel
	// AT_DISPATCH_FLOATING_TYPES -> float 32, float64
	// AT_DISPATCH_FLOATING_TYPES_AND_HALF -> float 32, float 64, float16
	// AT_DISPATCH_INTEGRAL_TYPES -> integral

	AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",([&] {
		trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
	feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
	points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
	feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
	);
	}));


}
