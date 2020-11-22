__global__ void reduce0(int* g_odata, int* g_idata, int n)
{
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
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

__global__ void reduce1(int* g_odata, int* g_idata, int n)
{
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
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

__global__ void reduce2(int* g_odata, int* g_idata, int n)
{
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();


    for (unsigned int s = blockDim.x/2; s > 0; s >>=1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
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
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = g_idata[i]+g_idata[i+blockDim.x];
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