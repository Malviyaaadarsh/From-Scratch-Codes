#include<iostream>
#include<vector>
#include<cuda_runtime.h> 
using namespace std;

#define CHECK_CUDA_ERR(call)\
do { \
  cudaError_t err = call; \
  if(err!=cudaSuccess){\
    cerr << "CUDA Error : "<<cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << endl; \
    exit(EXIT_FAILURE);\
    }\
} while(0)

__global__ void vector_addition(const float *A, const float *B, float *C, int N){
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensuring No thread overflow
    if(globalIdx < N){ 
        C[globalIdx] = A[globalIdx] + B[globalIdx]; //vector addition
    }
}


int main(){
    int n = 1'000'000; // vector size
    size_t bytes = n * sizeof(float); 
 
    cout<< " Vector Addition using CUDA for " << n << " elements \n";
    // Memory Allocation on Host CPU 
    vector<float> h_a(n,1.5f), h_b(n,2.0f), h_c(n,0.0f);
    
    // Memory Allocation on Device GPU
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_c, bytes));

    // Copying data from Host CPU to GPU Device 
    CHECK_CUDA_ERR(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    
    // Adding vectors on GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // ceil(n/threadsPerBlock) rounded up value 
    cout<< "Launching kernel with " << blocksPerGrid << " blocks of " << threadsPerBlock << " threads each\n";
    vector_addition<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    CHECK_CUDA_ERR(cudaGetLastError()); CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // Copying result from GPU Device to Host CPU
    CHECK_CUDA_ERR(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Result Verification
    for(int i=0; i<n; i++){
        if(h_c[i] != h_a[i] + h_b[i]){
            cerr << "Result verification failed at index " << i << endl;
            exit(EXIT_FAILURE);
        }
    }
    cout<< "Sample Result : " << h_a[0] << " + " << h_b[0] << " = " << h_c[0] << endl;
    cout << "Result verification passed!\n";
    // Freeing VRAM 
    CHECK_CUDA_ERR(cudaFree(d_a)); CHECK_CUDA_ERR(cudaFree(d_b)); CHECK_CUDA_ERR(cudaFree(d_c));
    return 0 ; 
}

/*
Notes : 

1. cudaMemcpy() is synchronous by default, means that cpu will wait until the complete transfer. 
2. Ceil division trick : (N + threads-1)/threads , round up the value of n/threadsPerBlock to ensure that all elements are processed.
3. (i<N) check in kernel is important to avoid thread overflow, as the number of threads may exceed the number of elements in the vector 
   if we round up blocks using ceil division trick.
4. threadsPerBlock should almost always be a multiple of 32 (the size of a hardware Warp). 256 (8*32) or 512 (16*32) are 
   standard optimal choices because they ensure no execution lanes in a warp are wasted.
5. In vector_addition, thread 0 accesses A[0], thread 1 accesses A[1], etc. Because consecutive threads access consecutive 
   memory addresses in global VRAM, the GPU memory controller combines these into a single 128-byte memory transaction.
   This is called Global Memory Coalescing.

Memory Flow : 
    Allocate CPU RAM for A,B,C ---> Alocate GPU VRAM (cudaMalloc) for A,B,C ---> Copy A,B from CPU RAM to VRAM (cudaMemcpy) 
    ---> Run Addition kernel on GPU ---> Copy C from VRAM to CPU RAM (cudaMemcpy) ---> Free VRAM (cudaFree).
*/