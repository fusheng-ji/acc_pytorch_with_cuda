#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
		// __global__ means you call data form CPUs and compute on GPUs
		// __host__ means you call data form CPUs and computer on GPUs
		// __device__ means you call data form GPUs and compute on GPUs
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
		const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
		torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
){
	// compute the corresponding index
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	const int f = blockIdx.y * blockDim.y + threadIdx.y; 
	// remove useless threads
	if (n>=feats.size(0) || f>=feats.size(2)) return;
	
	// point -1~1
	const scalar_t u = (points[n][0]+1)/2;
	const scalar_t v = (points[n][1]+1)/2;
	const scalar_t w = (points[n][2]+1)/2;

	const scalar_t a = (1-v)*(1-w);
	const scalar_t b = (1-v)*w;
	const scalar_t c = v*(1-w);
	const scalar_t d = 1-a-b-c;
	
	// compute interpolation and output
	feat_interp[n][f] = 
		(1-u)*(
			a*feats[n][0][f] +
			b*feats[n][1][f] +
			c*feats[n][2][f] +
			d*feats[n][3][f]
			)
	       	+ u*(
			a*feats[n][4][f] +
			b*feats[n][5][f] +
			c*feats[n][6][f] +
			d*feats[n][7][f]
			);
}

torch::Tensor trilinear_fw_cu( //fw means forward
	torch::Tensor feats,
	torch::Tensor points	
){
	const int N = feats.size(0), F = feats.size(2);
	torch::Tensor feat_interp = torch::zeros({N, F}, feats.options()); // if write in cpp
	// if write in python:
	// feat_interp = torch.zeros(N, F, dtype=torch.float32, device='cuda:1')
	// what if we want to define a variable which have different dtype with input, 
	//such as integer, with cpp:
	// torch::Tensor feat_interp = torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device()))
	const dim3 threads(16, 16); // define size of thread
	const dim3 blocks((N+threads.x-1)/threads.x,
			(F+threads.y-1)/threads.y); //define size of block
	// launch kernel
	// AT_DISPATCH_FLOATING_TYPES -> float32, float64
	// AT_DISPATCH_FLOATING_TYPES_AND_HALF -> float32, float64, float16
	// AT_DISPATCH_INTEGRAL_TYPES -> integral

	AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
	([&] {
		// function name<place_holder(dtype)><<<const, const>>>(input, output);
		trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
				// tensor.packed_accessor<dtype, dim of tensor, Flag, size>()
				// packed_accessor is called for dealing with data with type of tensor
			    	// torch::RestrictPtrTraits means each value is stored in different places
				feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
				points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
				feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
		);
	}));
}
