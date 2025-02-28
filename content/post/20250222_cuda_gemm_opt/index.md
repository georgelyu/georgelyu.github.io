---
title: CUDA GEMM 优化
description: CUDA 矩阵乘法的一步步优化指南
slug: cuda_gemm_opt
date: 2025-02-22 21:32:50+0800
# image: cover.jpg
categories:
    - CUDA
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

因为最近读完了《Programming Massively Parallel Processors: A Hands-on Approach (4th Edition)》这本书（下面简称 PMPP），非常想结合书中的内容实操一下。

所以我结合了 [cuda 入门的正确姿势：how-to-optimize-gemm](https://zhuanlan.zhihu.com/p/478846788) 和 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) 这两篇非常好的文章，自己结合 PMPP 中的经验写了一版代码，同时记录一下学习的过程。

## 第一版实现：naive

第一版基本是最 naive 的实现。如果这个矩阵乘法是 $A \times B = C$，$A$、$B$、$C$ 三个矩阵的维度分别为 $m \times k$、$k \times n$、$m \times n$，则下面的 kernel 中每个线程对应矩阵 $C$ 中的一个元素，所以每个线程会进行 $k$ 次浮点数乘加。

```cuda
#define A(i, j) d_A[(i) * k + (j)]
#define B(i, j) d_B[(i) * n + (j)]
#define C(i, j) d_C[(i) * n + (j)]

// Naive version
__global__ void MyMatMulKernel(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  int _m = blockIdx.y * blockDim.y + threadIdx.y;
  int _n = blockIdx.x * blockDim.x + threadIdx.x;

  if (_m < m && _n < n) {
    Value_t c_value = 0;
    for (int _k = 0; _k < k; ++_k) c_value += A(_m, _k) * B(_k, _n);
    C(_m, _n) = c_value;
  }
}

void MyMatMul(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  constexpr int kBlockDim = 16;

  dim3 block(kBlockDim, kBlockDim);
  dim3 grid((n + kBlockDim - 1) / kBlockDim, (m + kBlockDim - 1) / kBlockDim);

  MyMatMulKernel<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
```

因为是第一版实现，所以没有在乎任何的参数设置（如 block 大小等）。同时说一下下面所有的测试中都使用 `float` 类型计算，并用我自己的 NVIDIA RTX 3090 来测试。这一版的性能可以达到大约 2.2 TFLOPS（根据参数该卡的理论性能为 35.58 TFLOPS）。

并且注意到，这里的写法已经进行了 global memory 的 coalesced access。

## 第二版实现：block tiling

第二版使用了 PMPP 第 5.4 和 5.5 节中提到的 tiling 方法，即将会被重复使用的数据放置在 block 的 shared memory 中，这样减少了 global memory 的重复传输。这个思想也是对应原 repo 中的 [MMult_cuda_3.cu](https://github.com/tpoisonooo/how-to-optimize-gemm/blob/master/cuda/MMult_cuda_3.cu)。

```cuda
#define A(i, j) d_A[(i) * k + (j)]
#define B(i, j) d_B[(i) * n + (j)]
#define C(i, j) d_C[(i) * n + (j)]

// Tiled version
template <int kTileWidth>
__global__ void MyMatMulKernel(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  __shared__ Value_t Ads[kTileWidth][kTileWidth];
  __shared__ Value_t Bds[kTileWidth][kTileWidth];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int _m = by * kTileWidth + ty;
  int _n = bx * kTileWidth + tx;

  Value_t c_value = 0;
  /* k operations of multiply-add are divided into phases, each phase correspond to an
   * iteration of for-loop */
  for (int ph = 0; ph < std::ceil((Value_t)k / kTileWidth); ++ph) {
    /* Collectively load data into shared memory */
    if (_m < m)
      Ads[ty][tx] = A(_m, ph * kTileWidth + tx);
    else
      Ads[ty][tx] = 0;

    if (_n < n)
      Bds[ty][tx] = B(ph * kTileWidth + ty, _n);
    else
      Bds[ty][tx] = 0;

    // Make sure all threads in block finished loading data
    __syncthreads();

    /* Do multiple-add */
    for (int k = 0; k < kTileWidth; ++k) c_value += Ads[ty][k] * Bds[k][tx];

    // Make sure all threads in block finished using shared memory, so that we can go into
    // next iteration
    __syncthreads();
  }

  if (_m < m && _n < n) C(_m, _n) = c_value;
}

void MyMatMul(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  constexpr int kTileWidth = 16;

  dim3 block(kTileWidth, kTileWidth);
  dim3 grid((n + kTileWidth - 1) / kTileWidth, (m + kTileWidth - 1) / kTileWidth);

  MyMatMulKernel<kTileWidth><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
```

这里的写法大部分是直接用了 PMPP 书中 kernel 的写法。这一版可以达到大概 2.9 TFLOPS（小小进步）。

我们这里可以简单计算一下，每个线程的访存（只考虑读）次数：

- GMEM：K / kTileWidth \* 2 loads，代入数值（kTileWidth = 16）得到 K / 8 loads
- SMEM：K * 2 loads

这里因为每个线程输出一个结果元素，所以每个结果元素的访存次数和每个线程的访存次数是相等的。

## 第三版实现：thread coarsening

这一版就比较有意思了，我们先来说一下应用的优化手法。手法是 PMPP 第 6.3 节中提到的 thread coarsening，即增加每个线程的颗粒度，不要使一个线程只负责一个元素。因为这种很细的颗粒度会导致很多的线程，所以会导致很多的 block。而当 block 或线程数很多的时候，在 GPU 中 block 之间会开始串行执行，这就增加了运行的 overhead。

还是拿我的 RTX 3090 举例，当我们设置三个矩阵都是 $1024 \times 1024$ 的方阵时，如果我们用最细颗粒度和 $16 \times 16 = 256$ 的 block 大小（随意定的），那么我们会有 4096 个 block。

然而查表可知，RTX 3090 每个 SM 最多支持 1536 个线程，16 个 block。因为我们的 block 大小是 256 个线程，所以由于每个 SM 上最大线程数（1536）的限制，一个 SM 只会分到 $1536 \div 256 = 6$ 个 block。而 RTX 3090 一共只有 82 个 SM，所以总共的 4096 个 block 需要 $4096 \div 6 \div 82 \approx 8.33$ *轮*。这个轮在官方语言中称为 *waves*。在使用 Nsight Compute 进行 profiling 时，我们能看到这样一个属性就是“Waves per SM”，它的值也确实是我们计算的 8.33。

![在对第二版实现使用 Nsight Compute 进行 profiling 时的性能报告，其中高亮的是“Waves per SM”。](waves_per_sm.jpg)

可以想象这每一轮都会有并行计算的一些 overhead（如 block 的调度开销等），所以我们可以让一个线程多处理几个元素，从而减小 block 的大小，同时也减小了总线程数。这是我们的第三版实现，在第二版上增加了一个 `stride` 选项，控制一个线程会计算多少个元素。

```cuda
#define A(s, i, j) d_A[((i) + s * kTileWidthY) * k + (j)]
#define B(s, i, j) d_B[((i) + s * kTileWidthY) * n + (j)]
#define C(s, i, j) d_C[((i) + s * kTileWidthY) * n + (j)]

// Tiled version with thread coarsening
template <int kTileWidthX, int kTileWidthY, int kStrideY>
__global__ void MyMatMulKernel(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  __shared__ Value_t Ads[kTileWidthX][kTileWidthY * kStrideY];
  __shared__ Value_t Bds[kTileWidthX][kTileWidthY * kStrideY];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int _m = by * (kTileWidthY * kStrideY) + ty;
  int _n = bx * kTileWidthX + tx;

  Value_t c_value[kStrideY] = {0};
  /* k operations of multiply-add are divided into phases, each phase correspond to an
   * iteration of for-loop */
  for (int ph = 0; ph < std::ceil((Value_t)k / kTileWidthX); ++ph) {
    for (int s = 0; s < kStrideY; ++s) {
      /* Collectively load data into shared memory */
      if (s * kTileWidthY + _m < m)
        Ads[ty + s * kTileWidthY][tx] = A(s, _m, ph * kTileWidthX + tx);
      else
        Ads[ty + s * kTileWidthY][tx] = 0;

      if (_n < n)
        Bds[ty + s * kTileWidthY][tx] = B(s, ph * kTileWidthX + ty, _n);
      else
        Bds[ty + s * kTileWidthY][tx] = 0;
    }

    // Make sure all threads in block finished loading data
    __syncthreads();

    // if (by == 1 && bx == 0 && tx == 0 && ty == 0) printf("%f %f\n", _m, _n);

    for (int s = 0; s < kStrideY; ++s)
      for (int k = 0; k < kTileWidthX; ++k) /* Do multiple-add */
        c_value[s] += Ads[ty + s * kTileWidthY][k] * Bds[k][tx];

    // Make sure all threads in block finished using shared memory, so that we can go
    // into next iteration
    __syncthreads();
  }

  for (int s = 0; s < kStrideY; ++s)
    if (s * kTileWidthY + _m < m && _n < n) C(s, _m, _n) = c_value[s];
}

void MyMatMul(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  constexpr int kTileWidthX = 16;
  constexpr int kTileWidthY = 8;
  constexpr int kStrideY = 2;

  dim3 block(kTileWidthX, kTileWidthY);
  dim3 grid((n + kTileWidthX - 1) / kTileWidthX,
            (m + (kTileWidthY * kStrideY) - 1) / (kTileWidthY * kStrideY));

  MyMatMulKernel<kTileWidthX, kTileWidthY, kStrideY>
      <<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
```

在这份代码里，为了不显著地增加每个 block 的 shared memory 用量，我们将每个 block 实际负责的 tile 大小依然定为 $16 \times 16$，而 `kStrideY` 被用来控制每个线程负责的元素数量，从而控制了 block 中的线程大小。注意到每个 block 的大小现在是 `kTileWidthX * kTileWidthY`。而为了保证 tile 大小仍是 $16 \times 16$，我们要人为地保证 `kTileWidthY * kStrideY = 16`。所以显而易见地，如果我们将 `kStrideY` 设为 1，则它会回退到第二版实现；如果我们将`kStrideY` 设为 2、4、8 或 16，则可以进行我们上面讲的 thread coarsening。

在这一版中，由于我进行了上面的计算，所以我发现，如果 stride 超过 2，比如 stride 为 4 时，每个 block 的大小为 64 个线程，这样每个 block 的线程就会过小，使得 SM 填不满。上面我们提到，一个 SM 中只能驻 16 个 block，所以在这种情况下只能驻 $16 \times 64 = 1024$ 个线程。这会让 occupancy 掉至 $1024 \div 1536 \approx 66.7$%。所以这里我先令 stride 为 2，测试结果可以达到大约 4.1 TFLOPS（又一个小小进步）。

### 分析：stride 调参

这里当然既然有 stride 这个参数，不妨试一下各个参数值的效果。我测试了四个不同的参数值结果如下：

| stride               | 1      | 2      | 4      | 8      | 16     |
|----------------------|--------|--------|--------|--------|--------|
| 计算吞吐量（TFLOPS）  | 2.9    | 4.1    | 5.1    | 6.0    | 3.8    |
| Block 线程数         | 256    | 128    | 64     | 32     | 16     |
| 理论 Occupancy       | 100%   | 100%   | 66.67% | 33.33% | 33.33% |
| 实际 Occupancy       | 95.75% | 90.90% | 59.86% | 30.29% | 29.42% |

可以发现这里很“反逻辑”的一点是随着 occupancy 变低，效率反而越来越高，直到 stride 到达 16。stride 到达 16 时，一个 block 的线程数已经低于 32 了，而 warp 是以 32 个线程为一组的，所以很显然浪费了很多效率。但是在 stride 从 1 到 8 的时候效率是不断提升的。

为了研究这个问题的原因，我搜到了一个 talk《Better Performance at Lower Occupancy》，我阅读的笔记放在[这篇文章]({{< ref "/post/20250223_lower_occupancy_note/index.md" >}})里。

在这篇 talk 里，它的 case study 就是讲的 GEMM 的 thread coarsening，可以说和我这里的第三版的修改一模一样。在分析这个原因时，该 talk 中提到一个很重要的点，就是对于 GEMM 这样的 memory-intensive 的应用，计算单元的计算吞吐量是用不完的。RTX 3090 的一个 SM 的计算吞吐量是 $35.58\text{ TFLOPS } \div 82\text{ SMs }\approx 444.3\text{ GFLOPS}$。但是可以看到，我们这里的每一次乘加，都需要访问两个浮点数。所以要跑满这个计算吞吐量，我们需要 $444.3\text{ GFLOPS } \div 2\text{ ops } \times 8\text{ B } = 1777.2\text{ GB/s }$。即使我们已经把元素整体地预先 load 到 shared memory 里面了，然而 shared memory 的带宽是 $32 \text{ banks } * 4\text{ B/bank } * 1.395\text{ GHz }= 178.56\text{ GB/s}$。很明显，shared memory 的带宽依然是瓶颈。

回到我们这个例子，我们注意到在第三版代码中的第 44 行有一个 `Bds[k][tx]`，这个元素是会被重用的。所以这里就减少了 shared memory 的读写。

同样算一下每个线程的访存（只考虑读）次数：

- GMEM：K / kTileWidth \* stride \* 2 loads，代入数值（kTileWidth = 16，stride = 8）得到 K loads
- SMEM：K \* (stride + 1) loads，代入数值（kTileWidth = 16，stride = 8）得到 K \* 9 loads

但是一个线程现在输出 8 个元素了所以每个结果元素需要的访存量变为：

- GMEM：K / 8 loads
- SMEM：K * 9 / 8 loads

当然我们要验证一下我们的想法对不对，在 Nsight Compute 中我们还能得到下面的一些性能指标：

| stride                        | 1       | 2       | 4       | 8       | 16      |
|-------------------------------|---------|---------|---------|---------|---------|
| 每线程寄存器量                | 38      | 40      | 40      | 52      | 72      |
| shared memory load 语句执行量 | 83.89 M | 50.33 M | 33.55 M | 25.17 M | 41.94 M |
| FMA 利用率（% active cycle）  | 10.69   | 15.66   | 20.89   | 24.09   | 26.98   |

确实如我们所预料，shared memory 的加载数量在减小，同时 FMA 单元利用率在提升。

## 第四版实现：2D thread coarsening

那么我们基本已经确定要进一步提升计算吞吐量，所以我们要进一步减少访存量（这里说的访存是包含 shared memory 的）。所以我们进一步加强 thread coarsening 的强度，令一个线程计算 8 * 8 个元素。所以我们抛弃第三版中的 `kStride`，用二维的 `kThreadWorkDimX` 和 `kThreadWorkDimY`，表示一个 thread 的工作范围。

下面的实现参考了[这里](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh)的源码。

```c++
#define A(i, j) d_A[(i) * k + (j)]
#define B(i, j) d_B[(i) * n + (j)]
#define C(i, j) d_C[(i) * n + (j)]

// Tiled version with 2D thread coarsening
template <int kBlockWorkDimX, int kBlockWorkDimY, int kThreadWorkDimX,
          int kThreadWorkDimY, int kTileDimK>
__global__ void MyMatMulKernel(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  // Introduce kTileDimK, so the tile is not square, otherwise it would be too large
  __shared__ Value_t Ads[kBlockWorkDimY][kTileDimK];
  __shared__ Value_t Bds[kTileDimK][kBlockWorkDimX];

  const int by = blockIdx.y;
  const int bx = blockIdx.x;

  // (thread_outer_x, thread_outer_y) is the thread's outer position w.r.t. block.
  const int thread_outer_x = threadIdx.x % (kBlockWorkDimX / kThreadWorkDimX);
  const int thread_outer_y = threadIdx.x / (kBlockWorkDimX / kThreadWorkDimX);

  // Move tile to beginning of d_A's row and d_B's column
  d_A += by * kBlockWorkDimY * k;
  d_B += bx * kBlockWorkDimX;
  d_C += by * kBlockWorkDimY * n + bx * kBlockWorkDimX;

  // Calculate the indices that this thread will load into SMEM
  const int Ads_row = threadIdx.x / kTileDimK;
  const int Ads_col = threadIdx.x % kTileDimK;
  const int Bds_row = threadIdx.x / kBlockWorkDimX;
  const int Bds_col = threadIdx.x % kBlockWorkDimX;

  // The stride is for GMEM coalescing
  constexpr int thread_num_in_block =
      (kBlockWorkDimX * kBlockWorkDimY) / (kThreadWorkDimX * kThreadWorkDimY);

  const int strideA = thread_num_in_block / kTileDimK;
  const int strideB = thread_num_in_block / kBlockWorkDimX;

  // Register caches for Ads and Bds
  Value_t A_reg[kThreadWorkDimY] = {0.0};
  Value_t B_reg[kThreadWorkDimX] = {0.0};

  Value_t c_value[kThreadWorkDimY][kThreadWorkDimX] = {0};
  /* k operations of multiply-add are divided into phases, each phase correspond to an
   * iteration of for-loop */
  for (int ph = 0; ph < std::ceil((Value_t)k / kTileDimK); ++ph) {
    /* Collectively load data into shared memory */
    for (int loadOffset = 0; loadOffset < kBlockWorkDimY; loadOffset += strideA)
      Ads[Ads_row + loadOffset][Ads_col] = A(Ads_row + loadOffset, Ads_col);
    for (int loadOffset = 0; loadOffset < kTileDimK; loadOffset += strideB)
      Bds[Bds_row + loadOffset][Bds_col] = B(Bds_row + loadOffset, Bds_col);
    
    // Make sure all threads in block finished loading data
    __syncthreads();

    // Advance tile
    d_A += kTileDimK;      // move kTileDimK columns to right
    d_B += kTileDimK * n;  // move kTileDimK rows down

    // Calculate per-thread results
    for (int k = 0; k < kTileDimK; ++k) {
      // SMEM to registers
      for (int i = 0; i < kThreadWorkDimY; ++i)
        A_reg[i] = Ads[thread_outer_y * kThreadWorkDimY + i][k];
      for (int i = 0; i < kThreadWorkDimX; ++i)
        B_reg[i] = Bds[k][thread_outer_x * kThreadWorkDimX + i];

      for (int cy = 0; cy < kThreadWorkDimY; ++cy)
        for (int cx = 0; cx < kThreadWorkDimX; ++cx)
          c_value[cy][cx] += A_reg[cy] * B_reg[cx];
    }

    // Make sure all threads in block finished using shared memory, so that we can go
    // into next iteration
    __syncthreads();
  }

  // Write results to GMEM
  for (int cy = 0; cy < kThreadWorkDimY; ++cy)
    for (int cx = 0; cx < kThreadWorkDimX; ++cx)
      C(thread_outer_y * kThreadWorkDimY + cy, thread_outer_x * kThreadWorkDimX + cx) =
          c_value[cy][cx];
}

void MyMatMul(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  // This is LOGICAL, not REAL block dim.
  constexpr int kThreadWorkDimX = 8;
  constexpr int kThreadWorkDimY = 8;

  // kBlockWorkDimX/kThreadWorkDimX and kBlockWorkDimY/kThreadWorkDimY must be exact
  // division.
  constexpr int kBlockWorkDimX = 8 * kThreadWorkDimX;
  constexpr int kBlockWorkDimY = 8 * kThreadWorkDimY;

  constexpr int kTileDimK = 8;

  dim3 block((kBlockWorkDimX * kBlockWorkDimY) / (kThreadWorkDimX * kThreadWorkDimY));

  dim3 grid((n + kBlockWorkDimX - 1) / kBlockWorkDimX,
            (m + kBlockWorkDimY - 1) / kBlockWorkDimY);

  MyMatMulKernel<kBlockWorkDimX, kBlockWorkDimY, kThreadWorkDimX, kThreadWorkDimY,
                 kTileDimK><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
```

这个优化的关键在于第 60-70 行。通过 2D 的 thread coarsening，原理与 1D thread coarsening 类似，但是元素的重用可以更多。可以以 (kThreadWorkDimX + kThreadWorkDimY) 次 shared memory 加载进行 (kThreadWorkDimX * kThreadWorkDimY) 次乘加计算（图示过程参考[该文的 Kernel 5 部分](https://siboehm.com/articles/22/CUDA-MMM)）和[该文的第 1.3 节](https://zhuanlan.zhihu.com/p/441146275)。

这时每个线程需要的访存量（只考虑读）变为：

- GMEM：K / 8 (outer loop iters) \* 2 (A+B) \* 512/64 (sizeSMEM/numThreads) loads，为 K * 2 loads
- SMEM：K / 8 (outer loop iters) \* 8 (kTileDimK) \* (8 + 8) (kThreadWorkDimY + kThreadWorkDimX) loads，为 K * 16 loads

则每个结果元素需要的访存量（只考虑读）变为：

- GMEM：K / 32 loads
- SMEM：K / 2 loads

经过测试，这时性能已经来到了 15 TFLOPS，相比之前最高的 6 TFLOPS 可以说达到了质的飞跃。

## 第五版实现：vectorized memory access

这一版主要做了两个改变：

1. 在数据从 GMEM 读到 SMEM 时，把 As 进行转置，这样子可以进行 SMEM 的合并读取；
2. 在读写 GMEM 和 SMEM 时使用 `float4`，以使用向量化的指令（如 `lde.e.128`、`stg.e.128`、`lds.128`、`sts.128`等）

代码如下：

```c++
#define A(i, j) d_A[(i) * k + (j)]
#define B(i, j) d_B[(i) * n + (j)]
#define C(i, j) d_C[(i) * n + (j)]

// Tiled version with 2D thread coarsening and vectorized memory access
template <int kBlockWorkDimX, int kBlockWorkDimY, int kThreadWorkDimX,
          int kThreadWorkDimY, int kTileDimK>
__global__ void MyMatMulKernel(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  // Introduce kTileDimK, so the tile is not square, otherwise it would be too large
  // Here Ads is transposed
  __shared__ Value_t Ads_T[kTileDimK][kBlockWorkDimY];
  __shared__ Value_t Bds[kTileDimK][kBlockWorkDimX];

  const int by = blockIdx.y;
  const int bx = blockIdx.x;

  // (thread_outer_x, thread_outer_y) is the thread's outer position w.r.t. block.
  const int thread_outer_x = threadIdx.x % (kBlockWorkDimX / kThreadWorkDimX);
  const int thread_outer_y = threadIdx.x / (kBlockWorkDimX / kThreadWorkDimX);

  // Move tile to beginning of d_A's row and d_B's column
  d_A += by * kBlockWorkDimY * k;
  d_B += bx * kBlockWorkDimX;
  d_C += by * kBlockWorkDimY * n + bx * kBlockWorkDimX;

  // Calculate the indices that this thread will load into SMEM
  const int Ads_row = threadIdx.x / (kTileDimK / 4);
  const int Ads_col = threadIdx.x % (kTileDimK / 4);
  const int Bds_row = threadIdx.x / (kBlockWorkDimX / 4);
  const int Bds_col = threadIdx.x % (kBlockWorkDimX / 4);

  // The stride is for GMEM coalescing
  constexpr int thread_num_in_block =
      (kBlockWorkDimX * kBlockWorkDimY) / (kThreadWorkDimX * kThreadWorkDimY);

  const int strideA = thread_num_in_block / (kTileDimK / 4);
  const int strideB = thread_num_in_block / (kBlockWorkDimX / 4);

  // Register caches for Ads and Bds
  Value_t A_reg[kThreadWorkDimY] = {0.0};
  Value_t B_reg[kThreadWorkDimX] = {0.0};

  Value_t c_value[kThreadWorkDimY][kThreadWorkDimX] = {0};
  /* k operations of multiply-add are divided into phases, each phase correspond to an
   * iteration of for-loop */
  for (int ph = 0; ph < std::ceil((Value_t)k / kTileDimK); ++ph) {
    /* Collectively load data into shared memory */
    for (int loadOffset = 0; loadOffset < kBlockWorkDimY; loadOffset += strideA) {
      // Here A is transpoed and saved into SMEM
      float4 tmp = reinterpret_cast<float4*>(&A(Ads_row + loadOffset, Ads_col * 4))[0];
      Ads_T[Ads_col * 4 + 0][Ads_row + loadOffset] = tmp.x;
      Ads_T[Ads_col * 4 + 1][Ads_row + loadOffset] = tmp.y;
      Ads_T[Ads_col * 4 + 2][Ads_row + loadOffset] = tmp.z;
      Ads_T[Ads_col * 4 + 3][Ads_row + loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < kTileDimK; loadOffset += strideB)
      reinterpret_cast<float4*>(&Bds[Bds_row + loadOffset][Bds_col * 4])[0] =
          reinterpret_cast<float4*>(&B(Bds_row + loadOffset, Bds_col * 4))[0];

    // Make sure all threads in block finished loading data
    __syncthreads();

    // Advance tile
    d_A += kTileDimK;      // move kTileDimK columns to right
    d_B += kTileDimK * n;  // move kTileDimK rows down

    // Calculate per-thread results
    for (int k = 0; k < kTileDimK; ++k) {
      // SMEM to registers
      // Due to transposed A in SMEM, the row and col indicies are switched
      for (int i = 0; i < kThreadWorkDimY; ++i)
        A_reg[i] = Ads_T[k][thread_outer_y * kThreadWorkDimY + i];
      for (int i = 0; i < kThreadWorkDimX; ++i)
        B_reg[i] = Bds[k][thread_outer_x * kThreadWorkDimX + i];

      for (int cy = 0; cy < kThreadWorkDimY; ++cy)
        for (int cx = 0; cx < kThreadWorkDimX; ++cx)
          c_value[cy][cx] += A_reg[cy] * B_reg[cx];
    }

    // Make sure all threads in block finished using shared memory, so that we can go
    // into next iteration
    __syncthreads();
  }

  // Write results to GMEM
  for (int cy = 0; cy < kThreadWorkDimY; ++cy)
    for (int cx = 0; cx < kThreadWorkDimX; cx += 4)
      reinterpret_cast<float4*>(&C(thread_outer_y * kThreadWorkDimY + cy,
                                   thread_outer_x * kThreadWorkDimX + cx))[0] =
          reinterpret_cast<float4*>(&c_value[cy][cx])[0];
}

void MyMatMul(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  // This is LOGICAL, not REAL block dim.
  constexpr int kThreadWorkDimX = 8;
  constexpr int kThreadWorkDimY = 8;

  // kBlockWorkDimX/kThreadWorkDimX and kBlockWorkDimY/kThreadWorkDimY must be exact
  // division.
  constexpr int kBlockWorkDimX = 8 * kThreadWorkDimX;
  constexpr int kBlockWorkDimY = 8 * kThreadWorkDimY;

  constexpr int kTileDimK = 8;

  dim3 block((kBlockWorkDimX * kBlockWorkDimY) / (kThreadWorkDimX * kThreadWorkDimY));

  dim3 grid((n + kBlockWorkDimX - 1) / kBlockWorkDimX,
            (m + kBlockWorkDimY - 1) / kBlockWorkDimY);

  MyMatMulKernel<kBlockWorkDimX, kBlockWorkDimY, kThreadWorkDimX, kThreadWorkDimY,
                 kTileDimK><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
```

这一版的效率可以达到约 18 TFLOPS。

## 第六版实现：warp tiling

在第四版实现时我们说了，现在一个 thread tile 可以用 (kThreadWorkDimX + kThreadWorkDimY) 次 shared memory 加载进行 (kThreadWorkDimX * kThreadWorkDimY) 次乘加计算。所以很明显，在乘积确定时，kThreadWorkDimX 和 kThreadWorkDimY 越接近，算存比会越高。

那么对于一个 warp 的 32 个线程来说，这一过程是要进行 32 次的。我们可以认为每个 thread 在把数据从 SMEM 移至寄存器时，我们无形中是获得了 32 个 thread tile 在 warp 上。而这 32 个 thread tile 如果没有经过任何排布，比如是排成一行，那么可以想象，就和一个排成一行的 thread tile 一样，并没有最大化算存比。

所以通过使用 warp tile，即把 warp 也做类似的 coarsening，我们能做到的一个重要的事情是，我们可以手动控制这个 warp 的 thread tile 的分布。所以我们可以通过控制维度来让 warp 内 thread tile 的排列变成尽可能方形的，来最大化算存比。

同时我们注意到，在 kThreadWorkDimX = kThreadWorkDimY 时，这个值越大，算存比就越大。所以这样做给我们带来的另一个好处是，我们变相获得了一个很大的 thread tile。

![Warp tile 的示意图”。](warp_tiles.jpg)

代码如下：

```c++
#define A(i, j) d_A[(i) * k + (j)]
#define B(i, j) d_B[(i) * n + (j)]
#define C_interim(i, j) d_C_interim[(i) * n + (j)]

// Tiled version with 2D thread coarsening, vectorized memory access and 2D warp tiling
template <int kBlockWorkDimX, int kBlockWorkDimY, int kWarpWorkDimX, int kWarpWorkDimY,
          int kWarpTileWorkDimX, int kWarpTileWorkDimY, int kThreadWorkDimX,
          int kThreadWorkDimY, int kBlockTileDimK, int kThreadNumInBlock>
__global__ void MyMatMulKernel(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  // Introduce kTileDimK, so the tile is not square, otherwise it would be too large
  // Here Ads is transposed
  __shared__ Value_t Ads_T[kBlockTileDimK][kBlockWorkDimY];
  __shared__ Value_t Bds[kBlockTileDimK][kBlockWorkDimX];

  const int by = blockIdx.y;
  const int bx = blockIdx.x;

  constexpr int kWarpSize = 32;
  const int warpIdx = threadIdx.x / kWarpSize;  // the warp this thread is in

  const int wy = warpIdx / (kBlockWorkDimX / kWarpWorkDimX);
  const int wx = warpIdx % (kBlockWorkDimX / kWarpWorkDimX);

  // thread's position w.r.t. warp tile (ACTUAL warp).
  const int threadIdxInWarp = threadIdx.x % kWarpSize;
  const int ty_in_acutal_warp = threadIdxInWarp / (kWarpTileWorkDimX / kThreadWorkDimX);
  const int tx_in_acutal_warp = threadIdxInWarp % (kWarpTileWorkDimX / kThreadWorkDimX);

  // Move tile to beginning of d_A's row and d_B's column
  d_A += by * kBlockWorkDimY * k;
  d_B += bx * kBlockWorkDimX;
  // Here output position needs to go to warp output position
  d_C += (by * kBlockWorkDimY + wy * kWarpWorkDimY) * n + bx * kBlockWorkDimX +
         wx * kWarpWorkDimX;

  // Calculate the indices that this thread will load into SMEM
  const int Ads_row = threadIdx.x / (kBlockTileDimK / 4);
  const int Ads_col = threadIdx.x % (kBlockTileDimK / 4);
  const int Bds_row = threadIdx.x / (kBlockWorkDimX / 4);
  const int Bds_col = threadIdx.x % (kBlockWorkDimX / 4);

  const int strideA = kThreadNumInBlock / (kBlockTileDimK / 4);
  const int strideB = kThreadNumInBlock / (kBlockWorkDimX / 4);

  constexpr int kWarpTileNumY = kWarpWorkDimY / kWarpTileWorkDimY;
  constexpr int kWarpTileNumX = kWarpWorkDimX / kWarpTileWorkDimX;

  // Register caches for Ads and Bds (on the warptile level)
  Value_t A_reg[kWarpTileNumY][kThreadWorkDimY] = {0.0};
  Value_t B_reg[kWarpTileNumX][kThreadWorkDimX] = {0.0};

  // Results are in register file in warp scheduler
  Value_t c_value[kWarpTileNumY][kWarpTileNumX][kThreadWorkDimY][kThreadWorkDimX] = {0};

  /* k operations of multiply-add are divided into phases, each phase correspond to an
   * iteration of for-loop */
  for (int ph = 0; ph < std::ceil((Value_t)k / kBlockTileDimK); ++ph) {
    /* Collectively load data into shared memory */
    for (int loadOffset = 0; loadOffset < kBlockWorkDimY; loadOffset += strideA) {
      // Here A is transpoed and saved into SMEM
      float4 tmp = reinterpret_cast<float4*>(&A(Ads_row + loadOffset, Ads_col * 4))[0];
      Ads_T[Ads_col * 4 + 0][Ads_row + loadOffset] = tmp.x;
      Ads_T[Ads_col * 4 + 1][Ads_row + loadOffset] = tmp.y;
      Ads_T[Ads_col * 4 + 2][Ads_row + loadOffset] = tmp.z;
      Ads_T[Ads_col * 4 + 3][Ads_row + loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < kBlockTileDimK; loadOffset += strideB)
      reinterpret_cast<float4*>(&Bds[Bds_row + loadOffset][Bds_col * 4])[0] =
          reinterpret_cast<float4*>(&B(Bds_row + loadOffset, Bds_col * 4))[0];

    // Make sure all threads in block finished loading data
    __syncthreads();

    // Advance tile
    d_A += kBlockTileDimK;      // move kTileDimK columns to right
    d_B += kBlockTileDimK * n;  // move kTileDimK rows down

    // Calculate per-thread results
    for (int k = 0; k < kBlockTileDimK; ++k) {
      // SMEM to registers
      // Due to transposed A in SMEM, the row and col indicies are switched
      for (int w_sub_y = 0; w_sub_y < kWarpTileNumY; ++w_sub_y)
        for (int i = 0; i < kThreadWorkDimY; ++i)
          A_reg[w_sub_y][i] = Ads_T[k][wy * kWarpWorkDimY + w_sub_y * kWarpTileWorkDimY +
                                       ty_in_acutal_warp * kThreadWorkDimY + i];
      for (int w_sub_x = 0; w_sub_x < kWarpTileNumX; ++w_sub_x)
        for (int i = 0; i < kThreadWorkDimX; ++i)
          B_reg[w_sub_x][i] = Bds[k][wx * kWarpWorkDimX + w_sub_x * kWarpTileWorkDimX +
                                     tx_in_acutal_warp * kThreadWorkDimX + i];

      for (int w_sub_y = 0; w_sub_y < kWarpTileNumY; ++w_sub_y)
        for (int w_sub_x = 0; w_sub_x < kWarpTileNumX; ++w_sub_x)
          for (int cy = 0; cy < kThreadWorkDimY; ++cy)
            for (int cx = 0; cx < kThreadWorkDimX; ++cx)
              c_value[w_sub_y][w_sub_x][cy][cx] +=
                  A_reg[w_sub_y][cy] * B_reg[w_sub_x][cx];
    }

    // Make sure all threads in block finished using shared memory, so that we can go
    // into next iteration
    __syncthreads();
  }

  // Write results to GMEM
  for (int w_sub_y = 0; w_sub_y < kWarpTileNumY; ++w_sub_y)
    for (int w_sub_x = 0; w_sub_x < kWarpTileNumX; ++w_sub_x) {
      // move C pointer to current warp subtile
      Value_t* d_C_interim =
          d_C + (w_sub_y * kWarpTileWorkDimY) * n + w_sub_x * kWarpTileWorkDimX;

      for (int cy = 0; cy < kThreadWorkDimY; ++cy)
        for (int cx = 0; cx < kThreadWorkDimX; cx += 4)
          reinterpret_cast<float4*>(
              &C_interim(ty_in_acutal_warp * kThreadWorkDimY + cy,
                         tx_in_acutal_warp * kThreadWorkDimX + cx))[0] =
              reinterpret_cast<float4*>(&c_value[w_sub_y][w_sub_x][cy][cx])[0];
    }
}

void MyMatMul(float* d_A, float* d_B, float* d_C, int m, int n, int k) {
  // This is LOGICAL, not REAL block dim.
  constexpr int kNumOfThreads = 128;
  constexpr int kWarpSize = 32;
  constexpr int kNumOfWarps = kNumOfThreads / kWarpSize;

  constexpr int kThreadWorkDimX = 4;  // in elemenets
  constexpr int kThreadWorkDimY = 8;  // in elemenets

  // The work dim of one ACTUAL warp
  constexpr int kWarpTileWorkDimX = 4 * kThreadWorkDimX;  // 16 (in elemenets)
  constexpr int kWarpTileWorkDimY = kWarpSize / (kWarpTileWorkDimX / kThreadWorkDimX) *
                                kThreadWorkDimY;  // 64 (in elemenets)

  // The work dim of one LOGICAL warp
  constexpr int kWarpWorkDimX = 4 * kWarpTileWorkDimX;  // 64 (in elemenets)
  constexpr int kWarpWorkDimY = 1 * kWarpTileWorkDimY;  // 64 (in elemenets)

  constexpr int kBlockWorkDimX = 2 * kWarpWorkDimX;  // 128 (in elemenets)
  constexpr int kBlockWorkDimY = 2 * kWarpWorkDimY;  // 128 (in elemenets)

  constexpr int kBlockTileDimK = 16;

  static_assert(
      (kBlockWorkDimX % kWarpWorkDimX == 0) && (kBlockWorkDimY % kWarpWorkDimY == 0),
      "Block work dim is integral multiple of warp work dim.");
  static_assert(
      (kBlockWorkDimX / kWarpWorkDimX) * (kBlockWorkDimY / kWarpWorkDimY) == kNumOfWarps,
      "Number of warps times its work dim should be equal to block work dim.");
  static_assert((kWarpWorkDimY * kWarpWorkDimX) %
                        (kWarpSize * kThreadWorkDimY * kThreadWorkDimX * kNumOfWarps) ==
                    0,
                "Warp work dim is integral multiple of thread work dim.");
  static_assert(kWarpWorkDimX % kNumOfWarps == 0,
                "Warp work dim is multiple of NUM_WARP");
  static_assert((kNumOfThreads * 4) % kBlockTileDimK == 0,
                "NUM_THREADS*4 must be multiple of kTileDimK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((kNumOfThreads * 4) % kBlockWorkDimX == 0,
                "NUM_THREADS*4 must be multiple of kBlockWorkDimX to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(kBlockWorkDimX % (16 * kThreadWorkDimX) == 0,
                "kBlockWorkDimX must be a multiple of 16*kThreadWorkDimX to avoid "
                "quantization effects");
  static_assert(kBlockWorkDimY % (16 * kThreadWorkDimY) == 0,
                "kBlockWorkDimY must be a multiple of 16*kThreadWorkDimY to avoid "
                "quantization effects");
  static_assert((kBlockWorkDimY * kBlockTileDimK) % (4 * kNumOfThreads) == 0,
                "kBlockWorkDimY*kTileDimK must be a multiple of 4*kNumOfThreads to "
                "vectorize loads");
  static_assert((kBlockWorkDimX * kBlockTileDimK) % (4 * kNumOfThreads) == 0,
                "kBlockWorkDimX*kTileDimK must be a multiple of 4*kNumOfThreads to "
                "vectorize loads");

  dim3 block(kNumOfThreads);

  dim3 grid((n + kBlockWorkDimX - 1) / kBlockWorkDimX,
            (m + kBlockWorkDimY - 1) / kBlockWorkDimY);

  MyMatMulKernel<kBlockWorkDimX, kBlockWorkDimY, kWarpWorkDimX, kWarpWorkDimY,
                 kWarpTileWorkDimX, kWarpTileWorkDimY, kThreadWorkDimX, kThreadWorkDimY,
                 kBlockTileDimK, kNumOfThreads><<<grid, block>>>(d_A, d_B, d_C, m, n, k);
}
```