# Chapter 6

## Question 1

The kernels below are wasteful in their use of threads: half of the threads in each block never execute.

**Kernel 1 (naive, modular divergence):**
```cpp
__shared__ float partialSum[];
unsigned int t = threadIdx.x;
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (t % (2*stride) == 0)
        partialSum[t] += partialSum[t + stride];
}
```

**Kernel 2 (convergent):**
```cpp
__shared__ float partialSum[];
unsigned int t = threadIdx.x;
for (unsigned int stride = blockDim.x; stride > 1; stride /= 2) {
    __syncthreads();
    if (t < stride)
        partialSum[t] += partialSum[t + stride];
}
```

Modify the kernels to eliminate such waste. Give the relevant execution configuration parameter values at the kernel launch. Is there a cost in terms of an extra arithmetic operation needed? Which resource limitation can be potentially addressed with such a modification?

(Hint: line 2 and/or line 4 can be adjusted in each case; the number of elements in the section may increase.)

## Answer

**Core idea:** instead of one thread per element, use one thread per *two* elements. Each thread folds its two elements into a single value **once**, before the reduction loop starts. The loop itself is left unchanged.

**Kernel 1 (modular divergence version), modified:**
```cpp
__shared__ float partialSum[2*BLOCK_SIZE];
unsigned int t = threadIdx.x;
partialSum[t] += partialSum[t + blockDim.x];   // one-time fold, before reduction begins

for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (t % (2*stride) == 0)
        partialSum[t] += partialSum[t + stride];
}
```

**Kernel 2 (convergent version), modified:**
```cpp
__shared__ float partialSum[2*BLOCK_SIZE];
unsigned int t = threadIdx.x;
partialSum[t] += partialSum[t + blockDim.x];

for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (t < stride)
        partialSum[t] += partialSum[t + stride];
}
```

### Launch configuration

For a section of `SECTION_SIZE` elements, launch with:

```
blockDim.x = SECTION_SIZE / 2
```

— half the threads used by the original kernels, since each thread now owns two elements instead of one.

### Extra arithmetic cost

Yes — exactly **one** extra addition per thread (the fold-in line). This is a one-time, O(1) cost per thread; it does not grow with the number of loop iterations.

### Resource limitation addressed

The maximum number of threads per block (e.g. 1024 on many devices). Since each thread now owns two elements from the start, a single block can reduce twice as many elements for the same thread count — raising the largest section size one block can handle, and eliminating the population of threads that previously never executed.

### Note — a separate, related waste

Even after this fix, threads still go idle *across iterations* of the reduction loop itself (this is inherent to a tree reduction: active count halves each pass, N → N/2 → N/4 → ... → 1). This is not eliminable in the same way as the launch-time waste above. What differs between kernel 1 and kernel 2 is the *pattern* of that per-iteration idling: kernel 1's `t % (2*stride) == 0` scatters active threads (even `t` only), causing control divergence within a warp at every step. Kernel 2's `t < stride` keeps active threads contiguous starting from `0`, so divergence only occurs at a warp boundary — this is why kernel 2 is a distinct improvement over kernel 1, independent of the thread-count fix above.

## Question 2
Compare the modified kernels you wrote for Exercise 6.1. Which modification introduced fewer additional arithmetic operations?

## Answer
The same modification was applied to both kernels, so neither introduces fewer additional arithmetic operations than the other — they are equal.

Let the section contain `S` raw elements; each modified kernel launches `S/2` threads. Every one of these `S/2` threads performs exactly one addition, in parallel, before the reduction loop begins. Since the loop itself is unchanged, it still takes `log_2(S/2)` stages for the thread that survives to the end.

Thread 0's total number of stages is therefore:

`1 + log_2(S/2) = 1 + log_2(S) - 1 = log_2(S)`


## Question 3
Write a complete kernel based on Exercise 6.1 by (1) adding the statements that load a section of the input array from global memory to shared memory, (2) using blockIdx.x to allow multiple blocks to work on different sections of the input array, and (3) writing the reduction value for the section to a location according to the blockIdx.x so that all blocks will deposit their section reduction value to the lower part of the input array in global memory.

## Answer


```cpp
__global__
void reductionSum(float *dA)
{
    unsigned int t = threadIdx.x;
    unsigned int segment =  2 * blockDim.x * blockIdx.x;
    
    extern __shared__ float partialSum[];

    partialSum[t] = dA[segment + t];
    partialSum[t] += dA[segment + blockDim.x + t];

  
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }
    if (0 == t)
        dA[blockIdx.x] = partialSum[0];
}
```


Each block covers a section of `2 * blockDim.x` elements, so it's launched with dynamic shared memory sized accordingly:

```cpp
reductionSum<<<numBlocks, blockDim, 2 * blockDim.x * sizeof(float)>>>(dA, n);
```

# Question 4
Design a reduction program based on the kernel you wrote for Exercise 6.3. The host code should (1) transfer a large input array to
the global memory, and (2) use a loop to repeatedly invoke the kernel you wrote for Exercise 6.3 with adjusted execution configuration parameter values so that the reduction result for the input array will eventually be produced.

## Answer

```cpp
int main()
{

unsigned int n = 1e9;
float *hA = new float[n];
    for (unsigned int i = 0; i < n; i++)
        hA[i] = 1.0f;

float *dA;
int size = n * sizeof(float);

cudaMalloc((void **) &dA, size);
cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

unsigned int nThreadSize = 512;
dim3 dimBlock(nThreadSize,1,1);

unsigned int nNumberOfBlocks = ceil(n / (float)nThreadSize / 2);
    do
    {
        dim3 dimGrid(nNumberOfBlocks,1,1);
        reductionSum<<<dimGrid, dimBlock, 2 * blockDim.x * sizeof(float)>>>(dA);

        if(1 == nNumberOfBlocks)
            break;

        nNumberOfBlocks = ceil(nNumberOfBlocks / (float)nThreadSize / 2);

    }while(true)


// ==========================================
    //               VERIFICATION
    // ==========================================
    float h_result = 0.0f;
    
    // The final reduced sum will be at the 0th index of the device array
    cudaMemcpy(&h_result, dA, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate expected result mathematically
    float expected_result = (float)n;

    std::cout << "--- Reduction Sum Test ---" << std::endl;
    std::cout << "Expected Sum: " << expected_result << std::endl;
    std::cout << "GPU Actual Sum: " << h_result << std::endl;

    // Use a small epsilon for floating-point comparisons
    float epsilon = 1e-2;
    if (std::abs(h_result - expected_result) < epsilon) {
        std::cout << "[RESULT] TEST PASSED!" << std::endl;
    } else {
        std::cout << "[RESULT] TEST FAILED!" << std::endl;
    }
    // ==========================================




cudaFree(dA);
delete[] hA;
}
```

## Question 5
For the matrix multiplication kernel in Figure 6.11, draw the access patterns of threads in a warp of lines 9 and 10 for a small 16 x 16 matrix size. Calculate the tx and ty values for each thread in a warp and use these values in the d_M and d_N index calculations in lines 9 and 10. Show that the threads indeed access consecutive d_M and d_N locations in global memory during each iteration.

## Answer
Warp Memory Access Pattern — Tiled Matrix Multiplication (Lines 9 & 10)

**Assumptions:** 16×16 matrix, `TILE_WIDTH = 16` (whole matrix is one tile, so `Width/TILE_WIDTH = 1`, loop runs once with `m = 0`), single block (`bx = by = 0`), `blockDim = (16, 16)` → 256 threads = 8 warps.

 Thread-to-warp mapping

CUDA linearizes a 2D thread block with `threadIdx.x` as the fastest-varying index:

```
T = ty * blockDim.x + tx
```

With `blockDim.x = 16`, warp 0 (`T = 0..31`) is:

- `ty = 0`, `tx = 0..15`  (T = 0–15)
- `ty = 1`, `tx = 0..15`  (T = 16–31)

i.e. two full rows of `tx`, at `ty = 0` and `ty = 1`.

Line 9 — `d_M[Row*Width + m*TILE_WIDTH + tx]`

With `Row = by*TILE_WIDTH + ty, ty = 0,1`, `m = 0`, `Width = 16`:

```
addr_M = ty*16 + tx
```

| T | tx | ty | addr_M |
|---|----|----|--------|
| 0–15  | 0..15 | 0 | 0..15  |
| 16–31 | 0..15 | 1 | 16..31 |

Line 10 — `d_N[(m*TILE_WIDTH+ty)*Width + Col]`

With `Col = bx*TILE_WIDTH + tx, tx = 0...,15`, `m = 0`:

```
addr_N = ty*16 + tx
```

Same formula, same result:

| T | tx | ty | addr_N |
|---|----|----|--------|
| 0–15  | 0..15 | 0 | 0..15  |
| 16–31 | 0..15 | 1 | 16..31 |


```
d_M / d_N in global memory:
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 ... 31 ] ...
 └──────────── ty=0, tx=0..15 ───────────┘ └── ty=1, tx=0..15 ──┘
              (threads 0-15)                    (threads 16-31)
              ←──────────── all 32 threads of the warp ────────────→
```


For both `d_M` and `d_N`, warp 0 touches addresses **0 through 31 — one contiguous, monotonically increasing run**, with each thread reading exactly one distinct word. That's the definition of a coalesced access: the hardware can service the whole warp with one (or a couple of) 128-byte memory transactions instead of 32 separate ones.

The pattern generalizes: warp `k` (threads with `ty = 2k, 2k+1`) accesses addresses `32k` through `32k+31` — every warp reads a fresh, still-contiguous 32-word block, both on line 9 and line 10, in every iteration `m` of the loop (only the base offset `m*TILE_WIDTH` shifts, which doesn't break contiguity within a warp).

# Question 6
For the simple matrix-matrix multiplication (M X N) based on row-major layout, which input matrix will have coalesced accesses?
a. M
b. N
c. Both
d. Neither

# Answer
Within a warp, threadIdx.y (which determines Row) is constant, and threadIdx.x (which determines Col) is what varies across the warp. So during a given loop iteration (fixed k):
M[Row][k] — Row and k are both the same for every thread in the warp → every thread reads the identical address. That's a broadcast, not "different rows WIDTH apart." A broadcast is actually cheap (one memory transaction serviced to all threads), just not "coalesced" in the usual sense of 32 distinct-but-adjacent addresses.
N[k][Col] — k fixed, Col increments by 1 across the warp → truly consecutive addresses → coalesced.

# Question 7
For the tiled matrix-matrix multiplication (M X N) based on row-major layout, which input matrix will have coalesced accesses?
a. M
b. N
c. Both
d. Neither

# Answer
The reason M isn't coalesced is the broadcast pattern (same address for the whole warp), not threads hitting rows spaced WIDTH apart — that "rows WIDTH apart" description would only apply if you compared different warps (which map to different rows), which isn't what coalescing analysis is about. (b) N.

# Question 8
For the simple reduction kernel, if the block size is 1,024 and warp size is 32, how many warps in a block will have divergence during the fifth iteration?
a. 0
b. 1
c. 16
d. 32

# Answer
At fifth iteration stride is 2 * 16 <= 32 (warp size). There are 32 warps that will execute the if so (d) 32 warps will have divergence during the fifth iteration.

# Question 9
For the improved reduction kernel, if the block size is 1,024 and warp size is 32, how many warps will have divergence during the fifth iteration?
a. 0
b. 1
c. 16
d. 32

# Answer
In the improved reduction kernel, stride starts with block dimension which is 1024. In each iteration, 1024 -> 512 -> 256 -> 128 -> 64 -> 32. When stride is 32 only threads 0-31 will execute the if block and remaining threads will not execute. Since threads in the warps other than the first warp will not execute holly there will be no divergence. Also, since all threads in the first warp are smaller than 32 all of them will execute the if block so there will be no divergence. Thus, (a) 0 is the answer.

# Question 10
Write a matrix multiplication kernel function that corresponds to the design illustrated in Figure 6.12.

# Answer


```cpp
#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds0[TILE_WIDTH][2 * TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row  = by * TILE_WIDTH + ty;
    int Col0 = bx * (2 * TILE_WIDTH) + tx;   // left half
    int Col1 = Col0 + TILE_WIDTH;            // right half

    float Pvalue0 = 0.0f;
    float Pvalue1 = 0.0f;

    for (int m = 0; m < Width / TILE_WIDTH; ++m) {
        Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
        Nds0[ty][tx]              = d_N[(m * TILE_WIDTH + ty) * Width + Col0];
        Nds0[ty][tx + TILE_WIDTH] = d_N[(m * TILE_WIDTH + ty) * Width + Col1];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            float Mval = Mds[ty][k];
            Pvalue0 += Mval * Nds0[k][tx];
            Pvalue1 += Mval * Nds0[k][tx + TILE_WIDTH];
        }
        __syncthreads();
    }

    d_P[Row * Width + Col0] = Pvalue0;
    d_P[Row * Width + Col1] = Pvalue1;
}
```

# Question 11
The following scalar product code tests your understanding of the basic CUDA model. The following code computes 1,024 dot products, each of which is calculated from a pair of 256-element vectors. Assume that the code is executed on G80. Use the code to answer the following questions.

```cpp
 #define VECTOR_N 1024
 #define ELEMENT_N 256
 const int DATA_N = VECTOR_N * ELEMENT_N;
 const int DATA_SZ = DATA_N * sizeof(float);
 const int RESULT_SZ = VECTOR_N * sizeof(float);
...
 float d_A, d_B, d_C;
...
 cudaMalloc((void )&d_A, DATA_SZ);
 cudaMalloc((void )&d_B, DATA_SZ);
 cudaMalloc((void )&d_C, RESULT_SZ);
...
 scalarProd<<<VECTOR_N, ELEMENT_N>>>(d_C, d_A, d_B, ELEMENT_N);

 __global__ 
 void scalarProd(float d_C, float d_A, float d_B, int ElementN)
 {
 __shared__ float accumResult[ELEMENT_N];
 //Current vectors bases
 float A = d_A + ElementN * blockIdx.x;
 float B = d_B + ElementN * blockIdx.x;
 int tx = threadIdx.x;

 accumResult[tx] = A[tx] * B[tx];

 for(int stride = ElementN /2; stride > 0; stride >>= 1) // Line 23
 {
    __syncthreads();
    if(tx < stride)
        accumResult[tx] += accumResult[stride + tx];
 }
 d_C[blockIdx.x] = accumResult[0];
 }
```
a. How many threads are there in total?
b. How many threads are there in a warp?
c. How many threads are there in a block?
d. How many global memory loads and stores are done for each thread?
e. How many accesses to shared memory are done for each block?
f. List the source code lines, if any, that cause shared memory bank conflicts.
g. How many iterations of the for loop (line 23) will have branch divergence? Show your derivation.
h. Identify an opportunity to significantly reduce the bandwidth requirement on the global memory. How would you achieve this? How many accesses can you eliminate?

# ANSWER
a. 262,144 = 1024 * 256 threads.
b. 32.
c. 256.
d. 3.
e. 1,277 = 256 * 2 + (2^8 - 1) * 3 
f.  d_C[blockIdx.x] = accumResult[0] (However, broadcast exception: exact same address);
g. Warp size is 32. When stride becomes smaller than 32, there will be thread divergence for those warps. So starting 16 -> 8 -> 4 -> 2 -> 1. 5 iterations.
h. I would achive that by adding if (0 == t) to assigning d_C part. I would eliminate gridDim.x * (256 - 1) which is 261,120 accesses. Total stores: 262,144
Needed stores: 1024.


