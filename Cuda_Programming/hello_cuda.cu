/* 
My first CUDA program. Saying Hello to CUDA ( Compute Unified Device Architecture )

Compile the file using : nvcc hello_cuda.cu -o hello_cuda
Run the file using : ./hello_cuda

*/

#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// Each thread executes this kernel function
__global__ void hello_cuda() {
    int blockID = blockIdx.x;  // Current block index
    int threadID = threadIdx.x;  // Thread index within the current block
    int globalThreadID = blockID * blockDim.x + threadID; // Convert (block,thread) coordinates into a single linear thread ID // Linear thread numbering 
    
    printf("Hello CUDA from [Block %d, Thread %d] -> Global Thread ID: %d\n",blockID,threadID,globalThreadID);  // GPU side printing 
}
int main(){
    cout<<"Starting CUDA Host ...."<<endl;
    int blocks = 2; int threadsPerBlock = 4;
    cout << "Launching kernel with "<< blocks << " blocks, " << threadsPerBlock << " threads per block...\n\n";  // CPU side printing
    hello_cuda<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    cout << "\nGPU execution complete. Exiting Host program.\n";
    return 0;
}

/*
Notes : 
<<<blocks,threads>>> is the launch configuration for the kernel. It specifies how many blocks and how many threads per block will be used to execute the kernel on the GPU.
cudaDeviceSynchronize() is used to block the CPU until the GPU has completed all preceding requested tasks. This is important to ensure that the kernel has finished executing before the program continues.
__global__ : is a kernel function that runs on GPU and launches from CPU.
blockDim.x : is the number of threads in a block in the x dimension.

Note : 
1) The order of the printed lines is not guaranteed. GPU threads execute in parallel, so Block 1 may print before Block 0. 
2) printf() is supported inside CUDA kernels, std::count can't be used directly from GPU code.
*/


/*
Expected Output :

Starting CUDA Host ....
Launching kernel with 2 blocks, 4 threads per block...

Hello CUDA from [Block 1, Thread 0] -> Global Thread ID: 4
Hello CUDA from [Block 1, Thread 1] -> Global Thread ID: 5
Hello CUDA from [Block 1, Thread 2] -> Global Thread ID: 6
Hello CUDA from [Block 1, Thread 3] -> Global Thread ID: 7
Hello CUDA from [Block 0, Thread 0] -> Global Thread ID: 0
Hello CUDA from [Block 0, Thread 1] -> Global Thread ID: 1
Hello CUDA from [Block 0, Thread 2] -> Global Thread ID: 2
Hello CUDA from [Block 0, Thread 3] -> Global Thread ID: 3

GPU execution complete. Exiting Host program.
*/