#ifndef WARP_REDUCE_6
#define WARP_REDUCE_6
template <unsigned int blockSize>
__device__ void warpReduce6(volatile int* sdata, int tid)
{
    if (blockSize>=64) sdata[tid] += sdata[tid + 32];
    if (blockSize>=32) sdata[tid] += sdata[tid + 16];
    if (blockSize>=16) sdata[tid] += sdata[tid + 8];
    if (blockSize>= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize>= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize>= 2) sdata[tid] += sdata[tid + 1];
}
#endif // WARP_REDUCE_6

template <unsigned int blockSize>
__global__ void reduce6(int* g_odata, int* g_idata, int n)
{
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid]=0;

    while (i < n){sdata[tid] += g_idata[i]+g_idata[i+blockDim.x]; i+= gridSize;}
    __syncthreads();

    if (blockSize >= 1024){
        if (tid < 512) {sdata[tid] += sdata[tid+512];} __syncthreads();}
    if (blockSize >= 512){
        if (tid < 256) {sdata[tid] += sdata[tid+256];} __syncthreads();}
    if (blockSize >= 256){
        if (tid < 128) {sdata[tid] += sdata[tid+128];} __syncthreads();}
    if (blockSize >= 128){
        if (tid < 64) {sdata[tid] += sdata[tid+64];} __syncthreads();}

    if (tid < 32) warpReduce6<blockSize>(sdata,tid);

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];

}

void call_reduce6(int blocks, int threads, int smem_size, int* g_odata, int* g_idata, int n)
{
    // std::cout << "Use Reduction #7: Multiple Adds/Threads (reduce6)" << std::endl;
    switch(threads)
    {
        case 1024:
            reduce6<1024><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 512:
            reduce6< 512><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 256:
            reduce6< 256><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 128:
            reduce6< 128><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 64:
            reduce6<  64><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 32:
            reduce6<  32><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 16:
            reduce6<  16><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 8:
            reduce6<   8><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 4:
            reduce6<   4><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 2:
            reduce6<   2><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
        case 1:
            reduce6<   1><<< blocks, threads, smem_size>>>(g_odata, g_idata, n);
            break;
    }
    return;
}