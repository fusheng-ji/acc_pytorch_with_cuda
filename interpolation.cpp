#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor trilinear_interpolation(
		torch::Tensor feats,
		torch::Tensor point){
	return feats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("trilinear_interpolation", &trilinear_interpolation, R"pbdoc(
		m.def("name_in_python",&name_in_cpp)
		)pbdoc");
}
