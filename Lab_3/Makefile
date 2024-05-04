cuda_compile:
	nvcc cuda.cu -o cuda

nocuda_compile:
	g++ nocuda.cpp -o nocuda

stack_resize:
	ulimit -s $(s)
