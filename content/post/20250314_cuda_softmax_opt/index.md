---
title: CUDA Softmax 优化
description: CUDA softmax 函数的一步步优化指南
slug: cuda_softmax_opt
date: 2025-03-14 14:57:27+0800
# image: cover.jpg
categories:
    - CUDA
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

在这里记录一下关于 softmax 函数的 CUDA 实现的优化，基本翻译自 Maharshi Pandya 的 [Learning CUDA by optimizing softmax: A worklog](https://maharshi.bearblog.dev/optimizing-softmax-cuda/) 这篇博客。

## Softmax 的定义

Softmax 函数的输入是一个有 $N$ 个元素的数组 $X = \\{ x_i \\}$，输出是同样的一个有 N 个元素的数组 $O = \\{ o_i \\}$，第 $i$ 个输出元素 $o_i$ 的定义如下：

$$ o_i = \frac{e^{x_i}}{\sum_{k = 0}^{N - 1} e^{x_k}}. $$

这里能看到对于每个输入元素 $x_i$，在计算时除了其自身的值，主要还需要一个所有元素的指数和。

但是这里有一个问题，就是 $e^x$ 在 $x$ 较小时会很快地趋向于 0，在 $x$ 较大时则会很快地爆炸性增长。这对于浮点数的表示精度来说非常不好，即我们在使用 float 进行计算，且当 $x$ 有比较极端的值时，上面定义的 softmax 函数数值上并不稳定（分别会发生下溢和上溢）。

举例来说，对于 $X = \\{3, 1, -3\\}$，我们直接计算可以得到结果 \\O = {0.88, 0.12, 0\\}。但是对于 $X = \\{1000, 1000, 1000\\}$，我们会得到 -nan，因为使用 float 表示时，`exp(1000) = inf`。同理对于 $X = \\{-1000, -1000, -1000\\}$ 也是一样的会得到 -nan，因为 `-exp(1000) = 0`。

所以我们可以定义一个修改的 softmax 函数：把每个输入元素 $x_i$ 先减去数组中的最大值，即

$$ o_i = \frac{e^{x_i-x_{max}}}{\sum_{k = 0}^{N - 1} e^{x_k-x_{max}}}. $$

这样做的好处是保证了指数最大不会超过 0，所以不会发生上溢出（即不会得到 inf）。下溢出即使发生也没关系，下溢出的值被视为 0 不会影响我们得到一个合理的值。

当然最后我们证明一下，我们修改过的版本和之前是等价的：

$$ \begin{split} \frac{e^{x_i-x_{max}}}{\sum_{k = 0}^{N - 1} e^{x_k-x_{max}}} &= \frac{e^{-x_{max}} \cdot e^{x_i}}{e^{-x_{max}} \cdot \sum_{k = 0}^{N - 1} e^{x_k}} \\ &= \frac{e^{x_i}}{\sum_{k = 0}^{N - 1} e^{x_k}} \end{split} .$$

接下来我们在计算时都会使用这个经过修改的版本。

通常，在计算时，我们不会只对一个数组计算 softmax，而是对多个数组同时计算。我们假设有 $M$ 个这样的数组，每个数组有 $N$ 个元素，则我们有一个 $M \times N$ 的矩阵作为输入，同时我们的输出也是一个 $M \times N$ 的矩阵。

## 第一版实现：naive

接下来我们进行 CUDA 的实现。我们首先令一个线程处理一行数据，即对于一个数组进行一个串行的实现。这时我们需要三轮计算，因为每一轮都依赖于上一轮得到的结果：

1. 计算 $x_{max}$；
2. 计算 $\norm = \sum_{k = 0}^{N - 1} e^{x_k-x_{max}}$；
3. 计算 $o_i = \frac{e^{x_i-x_{max}}}{norm}$。

代码如下：

```c++
template <int kBlockDim>
__global__ void MySoftMaxKernel(float* d_X, float* d_O, int M, int N) {
  int row = blockIdx.x * kBlockDim + threadIdx.x;

  if (row < M) {
    // max
    float m = -1 * INFINITY;
    // norm factor
    float L = 0.0f;

    // 3 passes (not optimal)
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      m = max(m, d_X[i]);
    }
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      L += expf(d_X[i] - m);
    }
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      d_O[i] = expf(d_X[i] - m) / L;
    }
  }
}

void MySoftMax(float* d_X, float* d_O, int kM, int kN) {
  constexpr int kBlockDim = 1024;

  dim3 block(kBlockDim);
  dim3 grid((kM + kBlockDim - 1) / kBlockDim);

  MySoftMaxKernel<kBlockDim><<<grid, block>>>(d_X, d_O, kM, kN);
}
```

## 第二版实现：online softmax

我们先从算法角度进行一下优化，三轮显然计算有点多了，我们尝试能不能将第一轮（计算 $x_{max}$）和第二轮（计算 $norm$）进行融合。

因为我们是一个一个处理元素的，在处理过程中，$x_{max}$ 和 $norm$ 会不断地得到更新。我们先处理第一个元素 $x_0$，此时

- $x_{max0} = x_0$
- $norm_0 = e^{(x_0-x_{max0})}$

这时我们处理下一个元素 $x_1$，如果这个元素比 $x_{max0}$ 小的话，我们就不用修改 $x_{max}$，直接增加 $norm$ 即可，即

- $x_{max1} = x_{max0}$
- $norm_1 = norm_0 + e^{(x_1-x_{max1})}$

但如果 $x_1$ 比先前的最大值 $x_{max0}$ 大，则之前的 $norm$ 计算是有问题的，必须进行修正（因为 $x_{max}$ 更新了）。

这时我们对先前的 $norm0$ 乘一个修正项 $e^{(x_{max0} - x_{max1})}$，即可得到修正后的 $cnorm_0 = e^{(x_0-x_{max1})}$。所以这时我们得到了

- $x_{max1} = x_{1}$
- $norm_1 = norm_0 \cdot e^{(x_{max0} - x_{max1})} + e^{(x_1-x_{max1})}$

实际上这已经变成了一个递推式了，我们将这个递推式写成代码为：

```c++
template <int kBlockDim>
__global__ void MySoftMaxKernel(float* d_X, float* d_O, int M, int N) {
  int row = blockIdx.x * kBlockDim + threadIdx.x;

  if (row < M) {
    float m = -1 * INFINITY;
    float L = 0.0f;

    // compute max and norm factor in one pass only
    // by exploiting the property of exponentials
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      float curr = d_X[i];
      if (curr > m) {
        // norm needs to be mutiplied by correction term
        L = L * expf(m - curr);
        m = curr;
      }
      L += expf(curr - m);
    }
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      d_O[i] = expf(d_X[i] - m) / L;
    }
  }
}

void MySoftMax(float* d_X, float* d_O, int kM, int kN) {
  constexpr int kBlockDim = 1024;

  dim3 block(kBlockDim);
  dim3 grid((kM + kBlockDim - 1) / kBlockDim);

  MySoftMaxKernel<kBlockDim><<<grid, block>>>(d_X, d_O, kM, kN);
}
```

这样我们成功地省掉了一次 for 循环。

## 第三版实现：使用并行 reduction

先前的算法虽然节省了一个 for 循环，但显然对于单个数组（$M = 1$）的情况，我们的实现依然是串行的。然而从上面我们 online softmax 的推导可以得到，我们可以对一个数组分段地进行 $x_{max}$ 和 $norm$ 计算，再进行归约（reduce）。（在上面的推导中，我们可以认为是用一个多个元素得到的结果和一个单个元素进行了归约计算，但是这个可以很 trivial 地推广至多个元素和多个元素的归约计算。）

依据这个精神，我们可以用一个线程进行多个元素的 softmax 计算。为了保证读 global memory 的 coalescence，每个线程跨越 blockDim.x 去进行处理的，即线程 0 处理 {0, blockDim.x, blockDim.x * 2, ...}，线程 1 处理 {1, blockDim.x + 1, blockDim.x * 2 + 1, ...}。

在每个线程算完自己部分的局部 $x_{max}$ 和 $norm$ 之后，我们需要进行一个并行归约。这个过程每个线程会先把自己的局部结果存至 shared memory 之中，然后进行并行归约，最终结果会存在 `smem[0]` 之中。（这里我们假设了输入的数据规模用一个 block 就可以完成这个归约。）在对 $x_{max}$ 和 $norm$ 都完成归约之后我们就可以对结果进行并行地计算了。

参考代码如下。

```c++
template <int kBlockDim>
__global__ void MySoftMaxKernel(float* d_X, float* d_O, int M, int N) {
  // max and norm reduction will happen in shared memory (static)
  __shared__ float smem[kBlockDim];

  int row = blockIdx.x;
  int tid = threadIdx.x;

  // edge condition (we don't process further)
  if (row >= M) return;

  float* input_row = d_X + row * N;
  float* output_row = d_O + row * N;
  float local_max = -INFINITY;
  float local_norm = 0.0f;

  // compute local max and norm for each thread
  // and then finally have a sync barrier before moving on
  for (int i = tid; i < N; i += kBlockDim) {
    float x = input_row[i];
    if (x > local_max) {
      local_norm *= expf(local_max - x);
      local_max = x;
    }
    local_norm += expf(x - local_max);
  }
  __syncthreads();

  // each thread will have its own local max
  // we store it in the tid of the shared memory
  smem[tid] = local_max;
  __syncthreads();

  // block-level reduction in O(log(N)) time over all threads
  // is faster than linear reduction over all threads
  for (int stride = kBlockDim / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      smem[tid] = max(smem[tid], smem[tid + stride]);
    }
    // sync barrier before next iteration to ensure correctness
    __syncthreads();
  }

  // the first element after max reduction from all threads
  // will contain the global max for the row
  float row_max = smem[0];
  __syncthreads();

  // each thread will have its own local norm
  // we will store the corrected local norm in the shared memory
  // again, exploits property of exponentials
  smem[tid] = local_norm * expf(local_max - row_max);
  __syncthreads();

  // sum reduction similar to above for global norm factor
  for (int stride = kBlockDim / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  float row_norm = smem[0];
  __syncthreads();

  // finally, compute softmax
  for (int i = tid; i < N; i += kBlockDim) {
    output_row[i] = expf(input_row[i] - row_max) / row_norm;
  }
}

void MySoftMax(float* d_X, float* d_O, int kM, int kN) {
  constexpr int kBlockDim = 1024;

  dim3 block(kBlockDim);
  dim3 grid(kM);

  MySoftMaxKernel<kBlockDim><<<grid, block>>>(d_X, d_O, kM, kN);
}
```

## 第四版实现：warp-level reduction

我们注意到在上面的 reduction 中，我们使用了很多的 block 级的 `__syncthreads()`，同时我们还需要使用很多的 shared memory。为了避免这些开销，我们可以使用 warp-level primitives 来完成 reduction。

这些函数较为通用的写法是这样的：

```c++
/*
Takes in an array of size `TILE_SIZE` and reduces it as warp-wide sum.
The first element in the array will contain the reduced sum.
*/
__device__ __forceinline__ float warpReduceSum(float val) {
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
Takes in an array of size `TILE_SIZE` and reduces it warp-wide max.
The first element in the array will contain the reduced max.
*/
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
}
```

而在 kernel 中我们只需这样做：

```c++
float local_max = ...;
float max_reduce_result = warpReduceMax(local_max);

...

float local_norm = ...;
float norm_reduce_result = warpReduceSum(local_norm);
```

这样我们就完成了一个 warp 级的 reduction。不过在每个 warp 完成 reduction 之后，最终我们还是要对这些结果再进行一次 reduction。由于 warp-level reduction 的结果存在每个线程 0（warp 中编号）的寄存器中，所以想要让整个 block 拿到这个数据，我们还是要把这个结果放在 shared memory 中，然后再进行 reduction。如果这次的 reduction 的结果小于 32 个的话，我们甚至可以再进行一次 warp-level reduction。这里我们就做了这样的假设，连续使用了 warp-level reduction。

参考代码如下。

```c++
template <int kBlockDim>
__global__ void MySoftMaxKernel(float* d_X, float* d_O, int M, int N) {
  // number of threads in a warp
  constexpr int kWarpSize = 32;

  // max and norm reduction will happen in shared memory (static)
  __shared__ float smem[(kBlockDim + kWarpSize - 1) / kWarpSize];

  int row = blockIdx.x;
  int tid = threadIdx.x;

  if (row >= M) return;

  float* input_row = d_X + row * N;
  float* output_row = d_O + row * N;
  float local_max = -INFINITY;
  float local_norm = 0.0f;

  for (int i = tid; i < N; i += kBlockDim) {
    float x = input_row[i];
    if (x > local_max) {
      local_norm *= expf(local_max - x);
      local_max = x;
    }
    local_norm += expf(x - local_max);
  }
  __syncthreads();

  // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
  // note: if there are 256 threads in one block (8 warps of 32 threads each)
  // the following for loop reduces the value in all the 8 warps
  // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
  float val = local_max;
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }

  // when blockDim is greater than 32, we need to do a block level reduction
  // AFTER warp level reductions since we have the 8 maximum values that needs to be
  // reduced again the global max will be stored in the first warp
  if (kBlockDim > kWarpSize) {
    if (tid % kWarpSize == 0) {
      // which warp are we at?
      // store the value in its first thread index
      smem[tid / kWarpSize] = val;
    }
    __syncthreads();

    // first warp will do global reduction only
    // this is possible because we stored the values in the shared memory
    // so the threads in the first warp will read from it and then reduce
    if (tid < kWarpSize) {
      val = (tid < kBlockDim + (kWarpSize - 1) / kWarpSize) ? smem[tid] : -INFINITY;
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
      }
      if (tid == 0) smem[0] = val;
    }
  } else {
    // this is for when the number of threads in a block are not
    // greater than the warp size, in that case we already reduced
    // so we can store the value
    if (tid == 0) smem[0] = val;
  }
  __syncthreads();

  // we got the global row max now
  float row_max = smem[0];
  __syncthreads();

  // same reduction algorithm as above, but instead of max reduction
  // we do a sum reduction i.e. we accumulate the values
  val = local_norm * expf(local_max - row_max);
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }

  if (kBlockDim > kWarpSize) {
    if (tid % kWarpSize == 0) {
      smem[tid / kWarpSize] = val;
    }
    __syncthreads();

    // first warp will do global reduction
    if (tid < kWarpSize) {
      val = (tid < kBlockDim + (kWarpSize - 1) / kWarpSize) ? smem[tid] : 0.0f;
      for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
      }
      if (tid == 0) smem[0] = val;
    }
  } else {
    if (tid == 0) smem[0] = val;
  }
  __syncthreads();

  float row_norm = smem[0];
  __syncthreads();

  // finally, compute softmax
  for (int i = tid; i < N; i += kBlockDim) {
    output_row[i] = expf(input_row[i] - row_max) / row_norm;
  }
}

void MySoftMax(float* d_X, float* d_O, int kM, int kN) {
  constexpr int kBlockDim = 1024;

  dim3 block(kBlockDim);
  dim3 grid(kM);

  MySoftMaxKernel<kBlockDim><<<grid, block>>>(d_X, d_O, kM, kN);
}
```
