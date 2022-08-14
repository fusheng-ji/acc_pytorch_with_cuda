#include <torch/extension.h>
#include "utils.h"

torch::Tensor trilinear_interpolation(
		torch::Tensor feats,
		torch::Tensor points){
	CHECK_INPUT(feats);
	CHECK_INPUT(points);
	return trilinear_fw_cu(feats, points);
}

// input:
// 	feats: (N, 8, F)
// 	point: (N, 3)
// output:
// 	feat_interp: (N, F)
// tip: N and F can be computed parallel

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("trilinear_interpolation", &trilinear_interpolation, R"pbdoc(
		m.def("name_in_python",&name_in_cpp)
		)pbdoc");
}
