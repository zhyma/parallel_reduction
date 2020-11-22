#include <iostream>
#include <ctime>
#include <iomanip>
#include "reduce_0_3.h"
#include "reduce_4.h"
#include "reduce_5.h"
#include "reduce_6.h"

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
        case 0:
            reduce0<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 1:
            reduce1<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 2:
            reduce2<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 3:
            reduce3<<<blocks/2, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 4:
            reduce4<<<blocks/2, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 5:
            call_reduce5(blocks/2, threads, smem_size, g_odata, g_idata, n);
			break;
		case 6:
            call_reduce6(blocks/2, threads, smem_size, g_odata, g_idata, n);
            break;
        default:
            std::cout << "Not such a function! Error!" << std::endl;
            break;
    }
    return;
}

int main()
{
    std::clock_t start;
    double cpu_duration = 0;
    double gpu_duration = 0;
    int reduce = 5;

    int iter = 100;
    int len = 1 << 22;
    int* in;
    int* out;
    int blocksize = 1024;

	int gridsize = (len+blocksize-1)/blocksize;
	std::cout << "len/1024: " << len/1024 << std::endl;
	std::cout << "gridsize: " <<gridsize << std::endl;

    cudaMallocManaged(&in, sizeof(int) * len);
    cudaMallocManaged(&out, sizeof(int) * gridsize);
    for (int i = 0; i < len; ++i)
        in[i] = i;

    // gpu warm up
    gpu_sum(2, gridsize, blocksize, out, in, len);

    // CPU do the math
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    // Examine CPU time
    start = std::clock();
    // Call CPU sum here
    int cpu_out = cpu_sum(in, len);
    
    cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

    // GPU do the math
    int gpu_out = 0;
    // for(int k = 0; k < iter; ++k)
    // {
        //examine GPU time
        gpu_out = 0;
        for (int i = 0; i < gridsize; ++i)
            out[i] = 0;
        start = std::clock();
        // Call GPU sum here and sync
        
        gpu_sum(reduce, gridsize, blocksize, out, in, len);
        cudaDeviceSynchronize();
        
        // std::cout << "blocks: " << gridsize << std::endl;
		// TODO: gpu_sum(reduce, 1, blocksize,out, out, gridsize);
		int cnt = 0;
        for (int i = 0; i < gridsize; ++i)
        {
			gpu_out += out[i];
			if (out[i] != 0)
				cnt++;
				
		}
		std::cout <<  "non-zero block: " << cnt << std::endl;
        // std::cout << std::endl;
        gpu_duration += (std::clock() - start) / (double)CLOCKS_PER_SEC;
    // }

    if (cpu_out-gpu_out < 1 && gpu_out-cpu_out < 1)
    {
        std::cout << "wolf, tank, best match!" << std::endl;
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

    std::cout << "CPU time: " << cpu_duration << " s" << std::endl;
    std::cout << "GPU time: " << gpu_duration/iter << " s" << std::endl;

    cudaFree(in);

    return 0;
}