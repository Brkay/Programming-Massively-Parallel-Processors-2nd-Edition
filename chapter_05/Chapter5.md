# PMPP Chapter 5 — Shared Memory Exercises

## 5.1

**Question:** Consider the matrix addition in Exercise 3.1. Can one use shared memory to reduce the global memory bandwidth consumption?

*Hint: analyze the elements accessed by each thread and see if there is any commonality between threads.*

**Answer:** In matrix addition, `C[i][j] = A[i][j] + B[i][j]` — each input element is used exactly once, by exactly one thread, no matter how you partition the work (element/row/column). There is no data reuse to exploit, so there's no reduction in global memory bandwidth to be had from staging data through shared memory, regardless of thread granularity.

## 5.3

**Question:** What type of incorrect execution behavior can happen if one forgets to use `__syncthreads()` in the kernel of Figure 5.12?

**Answer:**
1. If we forgot the `__syncthreads()` at line 11: for a fast thread that executes the sum, a slow thread's shared memory can still hold values from a previous tile, since it hasn't finished updating `Mds`/`Nds` yet. In other words, a slow thread might not have updated the corresponding entries of `Mds` and `Nds`, so a wrong summation would happen.
2. If we forgot the `__syncthreads()` at line 14: a fast thread may already finish the summation and advance its `m` value in the outer `for` loop, overwriting shared memory for the next tile. A slow thread still computing with the old tile would then read corrupted values, since the fast thread has already updated the shared memory it's using.

## 5.4

**Question:** Assuming capacity was not an issue for registers or shared memory, give one case where it would be valuable to use shared memory instead of registers to hold values fetched from global memory. Explain your answer.

**Answer:** Registers are private to threads, so any value held there is only visible to that one thread. When multiple threads need to share the same data, holding it in shared memory (which is scoped to the block, not the thread) lets every thread in the block read it directly — avoiding redundant loads from global memory that registers alone would force each thread to make on its own.

## 5.5

**Question:** For our tiled matrix-matrix multiplication kernel, if we use a 32×32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?

a. 1/8 of the original usage
b. 1/16 of the original usage
c. 1/32 of the original usage
d. 1/64 of the original usage

**Answer:** The reduction in memory bandwidth is proportional to the tile dimension, since all 32 threads along a tile edge now share data amongst themselves instead of each re-fetching from global memory. Hence it is **1/32 (c)** of the original usage.

Here is the exact mathematical breakdown of why the bandwidth usage is reduced to 1/32 for a 32x32 tile.

Let $W$ be the width of the input matrices and $T$ be the tile width (where $T = 32$). We will compare the total global memory reads required by a single thread block (containing $T^2$ threads) to compute its assigned $T \times T$ section of the output matrix.

*   **Without Tiling (Naive):** Each thread independently computes one output element, requiring it to read $W$ elements from matrix M and $W$ elements from matrix N. For the entire block of $T^2$ threads, the total number of global memory reads is $2WT^2$.
*   **With Tiling:** The computation is broken into $W/T$ phases (looping over the tiles). In each phase, the block collaboratively loads one $T \times T$ tile from M and one $T \times T$ tile from N into shared memory. This takes $2T^2$ global reads per phase. The total global memory reads for the block across all phases is $2T^2(W/T) = 2WT$.

**The Reduction Ratio:**
$$\text{Reduction} = \frac{\text{Tiled Reads}}{\text{Naive Reads}} = \frac{2WT}{2WT^2} = \frac{1}{T}$$

By plugging in the specific tile width of $T = 32$, the bandwidth usage is reduced to exactly **1/32** of the original naive approach.

## 5.6

**Question:** Assume a kernel is launched with 1,000 thread blocks, each with 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the kernel's execution?

a. 1
b. 1,000
c. 512
d. 512,000

**Answer:** Local variables live in registers, which are private per thread. Hence `512 × 1000 = 512,000` versions are created **(d)**.

## 5.7

**Question:** In the previous question, if the variable is declared as a shared memory variable instead, how many versions will be created through the lifetime of the kernel's execution?

a. 1
b. 1,000
c. 512
d. 51,200

**Answer:** Shared memory variables are scoped per block, not per thread. Hence **1,000 (b)** versions are created — one per block.

## 5.8

**Question:** Explain the difference between shared memory and L1 cache.

**Answer:** Shared memory is explicitly managed by the programmer (software-controlled). L1 cache, by contrast, is managed by hardware/compiler heuristics — the developer has no direct control over what gets cached or evicted.

## 5.9

**Question:** Consider a matrix multiplication of two N×N input matrices. How many times is each element in the input matrices requested from global memory when:

a. There is no tiling?
b. Tiles of size T×T are used?

**Answer:** Each output element is a dot product of one row vector from the left matrix and one column vector from the right matrix.

a. Without tiling, each input element is re-requested once per output element that needs it — **N² times**.
b. With T×T tiling, each input element is loaded once per tile it participates in — **(N/T)² times**.

## 5.10

**Question:** A kernel performs 36 floating-point operations and 7 32-bit-word global memory accesses per thread. For each of the following device properties, indicate whether the kernel is compute- or memory-bound.

a. Peak FLOPS = 200 GFLOPS, peak memory bandwidth = 100 GB/s.
b. Peak FLOPS = 300 GFLOPS, peak memory bandwidth = 250 GB/s.

**Answer:**

Each thread does 36 FLOPs and 7 global memory accesses of 4 bytes each = 28 bytes.

Kernel's required ops-to-byte ratio: `36 / 28 ≈ 1.286 FLOP/byte`

To classify compute- vs. memory-bound, compare this to each device's own "machine balance" — `peak FLOPS ÷ peak bandwidth`, in FLOP/byte — the compute-to-memory throughput ratio the hardware itself is capable of.

**a. 200 GFLOPS, 100 GB/s**

Machine balance = `200 / 100 = 2.0 FLOP/byte`

The kernel needs only 1.286 FLOP/byte, but the hardware is built to supply 2.0 FLOP/byte — meaning for every byte it moves, it has more compute capacity available than this kernel actually uses. The kernel will exhaust memory bandwidth well before it saturates FLOPS capability.

→ **Memory-bound**

**b. 300 GFLOPS, 250 GB/s**

Machine balance = `300 / 250 = 1.2 FLOP/byte`

Now the hardware's balance point (1.2) is lower than what the kernel needs (1.286) — the kernel demands slightly more compute per byte than the device is naturally balanced to provide, so the compute units become the limiting factor before memory bandwidth is exhausted.

→ **Compute-bound**

## 5.11

**Question:** Indicate which of the following assignments per streaming multiprocessor is possible. Where not possible, indicate the limiting factor(s).

a. 4 blocks × 128 threads, 32 B shared memory/thread, compute capability 1.0.
b. 8 blocks × 128 threads, 16 B shared memory/thread, compute capability 1.0.
c. 16 blocks × 32 threads, 64 B shared memory/thread, compute capability 1.0.
d. 2 blocks × 512 threads, 32 B shared memory/thread, compute capability 1.2.
e. 4 blocks × 256 threads, 16 B shared memory/thread, compute capability 1.2.
f. 8 blocks × 256 threads, 8 B shared memory/thread, compute capability 1.2.

**Answer:**

**a.**
- Threads/SM: 4 × 128 = 512 ≤ 768 (CC 1.0 max) ✓
- Threads/block: 128 ≤ 512 ✓
- Blocks/SM: 4 ≤ 8 ✓
- Shared memory/SM: 4 × (32 B × 128) = 16,384 B = 16 KB ≤ 16 KB ✓ (exactly at the limit)

→ **Possible.**

**b.**
- Threads/block: 128 ≤ 512 ✓
- Blocks/SM: 8 ≤ 8 ✓
- Threads/SM: 8 × 128 = 1,024 vs. limit of 768 (CC 1.0) → ✗ exceeds
- Shared memory/SM: 16 B × 128 = 2,048 B/block × 8 = 16,384 B = 16 KB ≤ 16 KB ✓ (exactly at limit)

→ **Not possible.** Limiting factor: threads per SM (1,024 > 768).

**c.**
- Threads/block: 32 ≤ 512 ✓
- Threads/SM: 16 × 32 = 512 ≤ 768 ✓
- Blocks/SM: 16 vs. limit of 8 → ✗ exceeds
- Shared memory/SM: 64 B × 32 = 2,048 B/block × 16 = 32,768 B = 32 KB vs. limit of 16 KB → ✗ exceeds

→ **Not possible.** Two limiting factors: blocks per SM (16 > 8) and shared memory per SM (32 KB > 16 KB).

**d.**
- Threads/block: 512 ≤ 512 ✓ (exactly at limit)
- Blocks/SM: 2 ≤ 8 ✓
- Threads/SM: 2 × 512 = 1,024 ≤ 1,024 ✓ (exactly at limit, CC 1.2)
- Shared memory/SM: 32 B × 512 = 16,384 B/block × 2 = 32,768 B = 32 KB vs. limit of 16 KB → ✗ exceeds

→ **Not possible.** Limiting factor: shared memory per SM (32 KB > 16 KB).

**e.**
- Threads/block: 256 ≤ 512 ✓
- Blocks/SM: 4 ≤ 8 ✓
- Threads/SM: 4 × 256 = 1,024 ≤ 1,024 ✓ (exactly at limit)
- Shared memory/SM: 16 B × 256 = 4,096 B/block × 4 = 16,384 B = 16 KB ≤ 16 KB ✓ (exactly at limit)

→ **Possible.** All constraints satisfied — two of them (threads/SM and shared memory/SM) land exactly on the ceiling with no spare margin.

**f.**
- Threads/block: 256 ≤ 512 ✓
- Blocks/SM: 8 ≤ 8 ✓
- Threads/SM: 8 × 256 = 2,048 vs. limit of 1,024 (CC 1.2) → ✗ exceeds
- Shared memory/SM: 8 B × 256 = 2,048 B/block × 8 = 16,384 B = 16 KB ≤ 16 KB ✓

→ **Not possible.** Limiting factor: threads per SM (2,048 > 1,024).