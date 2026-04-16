all:
	nvcc \
	-O3 -c -o bicubic.o --resource-usage \
	`python3-config --includes` \
	`python3-config --ldflags --embed` \
	`python3 -c "from torch.utils.cpp_extension import include_paths;_=[print('-I'+x,end=' ') for x in include_paths('cuda')]"` \
	`python3 -c "from torch.utils.cpp_extension import library_paths;_=[print('-L'+x,end=' ') for x in library_paths('cuda')]"` \
	-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lcuda \
	-D_GLIBCXX_USE_CXX11_ABI=0 \
	bicubic.cu 


