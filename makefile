all:
	nvcc \
	-O3 --resource-usage \
	-I/usr/include/python3.12 -I/usr/include/python3.12 \
	-I/home/dchap/jaxenv/lib/python3.12/site-packages/torch/include \
	-I/home/dchap/jaxenv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
	-L/home/dchap/jaxenv/lib/python3.12/site-packages/torch/lib \
	-L/usr/lib/python3.12/config-3.12-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lpython3.12 -ldl  -lm \
	-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lcuda \
	-D_GLIBCXX_USE_CXX11_ABI=0 \
	bicubic.cu 