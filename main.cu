#include <iostream>
#include <ctime>
#include <iomanip>

__global__ void reduce0(float* g_odata, float* g_idata, int n)
{
	extern __shared__ float sdata[];

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

__global__ void reduce1(float* g_odata, float* g_idata, int n)
{
	extern __shared__ float sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n)?g_idata[i]:0;
	__syncthreads();


	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		unsigned int index = 2*s*tid;
		if (index < blockDim.x)
			sdata[index] += sdata[index + s];
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
	
}

float cpu_sum(float* h_in, float h_in_len)
{
	float total_sum = 0;
	for (int i = 0; i < h_in_len; ++i)
		total_sum = total_sum + h_in[i];

	return total_sum;
}

void gpu_sum(int whichKernel, int blocks, int threads, float* g_odata, float* g_idata, int n)
{
	int smem_size = (threads <= 32) ? 2*threads*sizeof(float) : threads * sizeof(float);
	// std::cout << "smem_size: " << smem_size << std::endl;
	switch (whichKernel)
	{
		case 0:
			reduce0<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
			break;
		case 1:
			reduce1<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
			break;
	}
	return;
}

int main()
{
	std::clock_t start;
	double cpu_duration, gpu_duration;

	// for (int pow2 = 5; pow2 < 16; ++pow2)
	for (int len = 5795; len < 5795+32; ++len)
	{
		// int len = 5795;
		// int len = 1 << pow2;
		// std::cout << std::endl << pow2 << std::endl;
		std::cout << std::endl << len << std::endl;
		float* in;
		float* out;
		int blocksize = 1024;
	
		cudaMallocManaged(&in, sizeof(float) * len);
		cudaMallocManaged(&out, sizeof(float) * (len+blocksize-1)/blocksize);
	
		for (int i = 0; i < len; ++i)
			in[i] = i;
	
		std::cout.setf(std::ios::fixed,std::ios::floatfield);
		// Examine CPU time
		start = std::clock();
		// Call CPU sum here
		float cpu_out = cpu_sum(in, len);
		
		cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		
	
		//examine GPU time
		start = std::clock();
		// Call GPU sum here and sync
		int gridsize = (len+blocksize-1)/blocksize;
		gpu_sum(0, gridsize, blocksize, out, in, len);
		cudaDeviceSynchronize();
		float gpu_out = 0;
		std::cout << "blocks: " << gridsize << std::endl;
		for (int i = 0; i < gridsize; ++i)
		{
			gpu_out += out[i];
			std::cout << out[i]/512 << ", ";
		}
		std::cout << std::endl;
		gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	
		if (cpu_out-gpu_out < 1 && gpu_out-cpu_out < 1)
		{
			std::cout << "wolf, tank, best match!" << std::endl;
			// std::cout << "CPU time: " << duration << " s" << std::endl;
			// std::cout << "GPU time: " << duration << " s" << std::endl;
		}
		else
		{
			// std::cout << "cpu out: " << cpu_out << std::endl;
			// std::cout << "gpu out: " << gpu_out << std::endl;
			std::cout << gpu_out-cpu_out << std::endl;
			// break;
		}

		cudaFree(in);
	}


	return 0;
}