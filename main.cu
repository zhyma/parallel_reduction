#include <iostream>
#include <ctime>
#include <iomanip>
#include <string>
#include <fstream>

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
            // std::cout << "Use Reduction #1: Interleaved Addressing (reduce0)" << std::endl;
            reduce0<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 1:
            // std::cout << "Use Reduction #2: Interleaved Addressing (reduce1)" << std::endl;
            reduce1<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 2:
            // std::cout << "Use Reduction #3: Sequential Addressing (reduce2)" << std::endl;
            reduce2<<<blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 3:
            // std::cout << "Use Reduction #4: First Add During Load (reduce3)" << std::endl;
            reduce3<<<blocks/2, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 4:
            // std::cout << "Use Reduction #5: Unroll the Last Warp (reduce4)" << std::endl;
            reduce4<<<blocks/2, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 5:
            call_reduce5(blocks/2, threads, smem_size, g_odata, g_idata, n);
			break;
		case 6:
            call_reduce6(blocks, threads, smem_size, g_odata, g_idata, n);
            break;
        default:
            std::cout << "Not such a function! Error!" << std::endl;
            break;
    }
    return;
}

int main()
{
    std::clock_t start, stop;
    double cpu_duration = 0;
    double gpu_duration = 0;

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

    // CPU do the math
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    // Examine CPU time
    start = std::clock();
    // Call CPU sum here
    int cpu_out = cpu_sum(in, len);
    
    cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "CPU time: " << cpu_duration << " s" << std::endl;

    for (int reduce = 0; reduce < 7; ++reduce)
    {
        std::cout << "reduce" << reduce << ", ";
        gpu_duration = 0;
        // GPU do the math
        int gpu_out = 0;

        //for reduction 1~6, the # of threads is fixed to 1024, the # of blocks is calculated
        if (reduce < 6)
        {
            for(int k = 0; k < iter+1; ++k)
            {
                // The first run is a warmup, discard.
                //examine GPU time
                gpu_out = 0;
                for (int i = 0; i < gridsize; ++i)
                    out[i] = 0;
                start = std::clock();
                // Call GPU sum here and sync
                
                gpu_sum(reduce, gridsize, blocksize, out, in, len);
                cudaDeviceSynchronize();
                
                for (int i = 0; i < gridsize; ++i)
                {
                    gpu_out += out[i];
                }
                stop = std::clock();
                
                if (abs(cpu_out-gpu_out) >= 1)
                {
                    std::cout << "ERROR!!! ";
                    std::cout << gpu_out-cpu_out << std::endl;
                }

                if (k > 0)
                    gpu_duration += (stop - start) / (double)CLOCKS_PER_SEC;
            }
        }
        else
        {
            // save reduction7's performance
            double gpu_best = 10e6;
            std::string best_config = "";
            std::ofstream out_file;
            out_file.open("reduction7_time.csv", std::ios::out);
            out_file << std::setiosflags(std::ios::fixed) << std::setprecision(6);

            reduce = 6;
            out_file <<" ,";
            for (gridsize = 32; gridsize <=(len+32-1)/32; gridsize = gridsize *2)
            {
                out_file << gridsize << ",";
            }
            out_file << std::endl;
            for (blocksize = 32; blocksize <= 1024; blocksize = blocksize * 2)
            {
                out_file << blocksize << ",";
                for (gridsize = 32; gridsize <=(len+32-1)/32; gridsize = gridsize *2)
                {
                    // the total # of threads larger than the # of elements to add
                    if (gridsize > (len+blocksize-1)/blocksize)
                    {
                        out_file << "-1,";
                        continue;
                    }

                    gpu_duration = 0;
                    // GPU do the math
                    gpu_out = 0;
                    for(int k = 0; k < iter+1; ++k)
                    {
                        // The first run is a warmup, discard.
                        //examine GPU time
                        gpu_out = 0;
                        for (int i = 0; i < gridsize; ++i)
                            out[i] = 0;
                        start = std::clock();
                        // Call GPU sum here and sync
                        
                        gpu_sum(reduce, gridsize, blocksize, out, in, len);
                        cudaDeviceSynchronize();
                        
                        int cnt = 0;
                        for (int i = 0; i < gridsize; ++i)
                        {
                            gpu_out += out[i];
                            if (out[i] != 0)
                                cnt++;
                        }
                        stop = std::clock();
                        
                        if (abs(cpu_out-gpu_out) >= 1)
                        {
                            std::cout << "ERROR!!! ";
                            std::cout << gpu_out-cpu_out << std::endl;
                        }

                        if (k > 0)
                            gpu_duration += (stop - start) / (double)CLOCKS_PER_SEC;
                    }

                    out_file << (int) (gpu_duration*1e6/iter) << ",";
                    if (gpu_duration < gpu_best)
                    {
                        gpu_best = gpu_duration;
                        best_config = "<<<" + std::to_string(gridsize) + ", " + std::to_string(blocksize) + ">>>";
                    }

                }
                out_file << std::endl;
            }
            out_file.close();
            gpu_duration = gpu_best;
            std::cout << "config to: " << best_config << std::endl;
        }

        if (cpu_out-gpu_out < 1 && gpu_out-cpu_out < 1)
        {
            std::cout << "wolf, tank, best match!" << std::endl;
        }

        std::cout << "GPU time: " << gpu_duration/iter << " s" << std::endl;
    }

    cudaFree(in);
    cudaFree(out);

    return 0;
}