/*
About thread indexing in CUDA programming : 

Physical threads are arranged into multidimensional blocks and grids. VRAM is always a linear array. 
So, the main goal of thread indexing is to map the multidimensional thread index to a linear index. 

Compile using :   nvcc thread_indexing.cu -o thread_indexing
Run using: ./thread_indexing
*/

#include<iostream>
#include<cuda_runtime.h>
using namespace std; 

__global__ void one_D_indexing(){                                 // Kernel function for one dimensional indexing
  int globalId = blockIdx.x * blockDim.x + threadIdx.x;
  printf("[Block %d, Thread %d] -> Global Thread ID: %d\n",blockIdx.x,threadIdx.x,globalId);
}

__global__ void two_D_indexing(){                                 // kernel function for two dimensional indexing
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int gridWidth = gridDim.x * blockDim.x ; 
    int globalId = row * gridWidth + col;                     // Flattening 2D coordinates into a single linear index.
    printf("[Block (%d,%d), Thread (%d,%d)]-> Global(x= %d,y= %d) -> Global Thread ID: %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,col,row,globalId);
}

int main(){
    cout<< " One Dimensional Indexing \n"; 
    int blocks_for_1D = 2 ; int threads_per_block_for_1D = 4 ;
    one_D_indexing<<<blocks_for_1D,threads_per_block_for_1D>>>();
    cudaDeviceSynchronize();

    cout << " Two Dimensional Indexing \n";
    dim3 threads_per_block_for_2D(2,2); dim3 blocks_for_2D(2,2);
    two_D_indexing<<<blocks_for_2D,threads_per_block_for_2D>>>();
    cudaDeviceSynchronize();

    cout<<" Execution Successful \n"; return 0 ; 
}

/*
Notes : 

1. The global thread ID is calculated by combining the block index, block dimension, and thread index.
2. In 2D indexing, the row and column coordinates are flattened into a single linear index.
3. dim3 is a CUDA data type used to define 1D, 2D, or 3D dimensions. 

Variables : 
threadIdx : local ID of thread within a block
blockIdx : block index in a grid
blockDim : number of threads in a block
gridDim : number of blocks in a grid
*/ 
