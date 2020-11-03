#include <iostream>
#include <ctime>
#include <iomanip>

__global__ void reduce1(int* g_odata, int* g_idata, int n)
{
	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n)?g_idata[i]:0;
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		if (tid % (2 * s) == 0)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
}

__global__ void reduce2(int* g_odata, int* g_idata, int n)
{
	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n)?g_idata[i]:0;
	__syncthreads();


	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		int index = 2*s*tid;
		if (index < blockDim.x)
			sdata[index] += sdata[index + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
}

__global__ void reduce3(int* g_odata, int* g_idata, int n)
{
	extern __shared__ int sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n)?g_idata[i]:0;
	__syncthreads();


	for (unsigned int s = blockDim.x/2; s > 0; s >>=1)
	{
		if (tid < s)
			sdata[tid] += sdata[tid+s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
}

int cpu_sum(int* h_in, int h_in_len)
{
	int total_sum = 0;
	for (int i = 0; i < h_in_len; ++i)
		total_sum = total_sum + h_in[i];

	return total_sum;
}

void gpu_sum(int whichKernel, int blocks, int threads, int* g_odata, int* g_idata, int n)
{
	int smem_size = (threads <= 32) ? 2*threads*sizeof(int) : threads * sizeof(int);
	// std::cout << "smem_size: " << smem_size << std::endl;
	switch (whichKernel)
	{
		case 1:
			reduce1<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
			break;
		case 2:
			reduce2<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
			break;
		case 3:
			reduce3<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
			break;
	}
	return;
}

int main()
{
	std::clock_t start;
	double cpu_duration = 0;
	double gpu_duration = 0;
	int reduce = 3;

	int iter = 40;
	for(int k = 0; k < iter; ++k)
	{
		int len = 1 << 22;
		int* in;
		int* out;
		int blocksize = 1024;

		int gridsize = (len+blocksize-1)/blocksize;

		cudaMallocManaged(&in, sizeof(int) * len);
		cudaMallocManaged(&out, sizeof(int) * gridsize);

		for (int i = 0; i < len; ++i)
			in[i] = i;

		std::cout.setf(std::ios::fixed,std::ios::floatfield);
		// Examine CPU time
		start = std::clock();
		// Call CPU sum here
		int cpu_out = cpu_sum(in, len);
		
		cpu_duration += (std::clock() - start) / (double)CLOCKS_PER_SEC;
		

		//examine GPU time
		start = std::clock();
		// Call GPU sum here and sync
		
		gpu_sum(reduce, gridsize, blocksize, out, in, len);
		cudaDeviceSynchronize();
		int gpu_out = 0;
		// std::cout << "blocks: " << gridsize << std::endl;
		// TODO: gpu_sum(reduce, 1, blocksize,out, out, gridsize);
		for (int i = 0; i < gridsize; ++i)
		{
			gpu_out += out[i];
		}
		// std::cout << std::endl;
		gpu_duration += (std::clock() - start) / (double)CLOCKS_PER_SEC;


		if (cpu_out-gpu_out < 1 && gpu_out-cpu_out < 1)
		{
			// std::cout << "wolf, tank, best match!" << std::endl;
			// std::cout << "CPU time: " << cpu_duration << " s" << std::endl;
			// std::cout << "GPU time: " << gpu_duration << " s" << std::endl;
		}
		else
		{
			// std::cout << "cpu out: " << cpu_out << std::endl;
			// std::cout << "gpu out: " << gpu_out << std::endl;
			std::cout << "ERROR!!! ";
			std::cout << gpu_out-cpu_out << std::endl;
			// break;
		}

		cudaFree(in);
	}
	std::cout << "CPU time: " << cpu_duration/iter << " s" << std::endl;
	std::cout << "GPU time: " << gpu_duration/iter << " s" << std::endl;

	return 0;
}