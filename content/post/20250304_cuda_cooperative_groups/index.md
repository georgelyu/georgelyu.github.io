---
title: 《Cooperative Groups Flexible CUDA Thread Programming》笔记
description: 更方便、细粒度的线程协作
slug: cuda_cooperative_groups
date: 2025-03-04 14:26:36+0800
# image: cover.jpg
categories:
    - CUDA
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

这是[《Cooperative Groups: Flexible CUDA Thread Programming》](https://developer.nvidia.com/blog/cooperative-groups/)这篇博客的学习笔记。

## 动机

在 CUDA 中线程之间分享数据和协作工作是非常常见的。CUDA 为此提供了一个同步函数 `__syncthreads()`，但是这个函数只能在 block 间同步。有时我们会需要更细粒度的线程协作。

所以 CUDA 推出了 Cooperative Groups programming model，这可以认为是原先 CUDA programming model 的一个扩展。

## Cooperative Groups 基础

使用 Cooperative Groups 需要加头文件 `#include <cooperative_groups.h>`，并且所有的命名都在 `cooperative_groups::` 命名空间下。

Cooperative Groups 中的基础类型是 `thread_group`，这是一个指向一组线程的 handle，这个 handle 只能被该组的线程访问。一个 group 有一些简单的接口，如 `unsigned size()` 来查询 group 内的线程数量，`unsigned thread_rank()` 来查询当前线程在 group 中的 id（在 `0` 到 `size() - 1` 之间）等。

对于一个 group，可以用下面的语句来同步。

```c++
g.sync();           // synchronize group g
cg::synchronize(g); // an equivalent way to synchronize g
```

## 创建 group

很显然，我们不用自己创建，block 本身就符合一个 group 概念。所以我们可以通过

```c++
thread_block block = this_thread_block();
```

来拿到指向该 block 的 handle。我们对这个 group 同步的话就和之前的 `__syncthreads()` 是一样的，所以下面的所有语句作用是相同的。

```c++
__syncthreads();
block.sync();
cg::synchronize(block);
this_thread_block().sync();
cg::synchronize(this_thread_block());
```

`thread_block` 相比上面的 `thread_group`，多了

```c++
dim3 group_index();  // 3-dimensional block index within the grid
dim3 thread_index(); // 3-dimensional thread index within the block
```

这两个值，等同于先前的 `blockIdx` 和 `threadIdx`。

想要把 group 继续细分，则可以使用 `cg::tiled_partition()` 函数，如我们可以用下面的代码把整个 block 分为 32 个线程的块，然后再分为 4 个线程一组的块：

```c++
thread_group tile32 = cg::tiled_partition(this_thread_block(), 32);
thread_group tile4 = tiled_partition(tile32, 4);
```

下面是一个 reduce sum 的例子：

```c++
#include <cooperative_groups.h>

using namespace cooperative_groups;
__device__ int reduce_sum(thread_group g, int *temp, int val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane<i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__device__ int thread_sum(int *input, int n) 
{
    int sum = 0;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n / 4; 
        i += blockDim.x * gridDim.x)
    {
        int4 in = ((int4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

__global__ void sum_kernel_32(int *sum, int *input, int n)
{
    int my_sum = thread_sum(input, n); 

    extern __shared__ int temp[];

    auto g = this_thread_block();
    auto tileIdx = g.thread_rank() / 32;
    int* t = &temp[32 * tileIdx];
    
    auto tile32 = tiled_partition(g, 32);  
    int tile_sum = reduce_sum(tile32, t, my_sum);

    if (tile32.thread_rank() == 0) atomicAdd(sum, tile_sum);
}
```

同时，对于 warp 来说，一个 warp 内的线程可能会发生 diverge，即 warp divergence。这时 SM 会用 active masks 来屏蔽没有激活的线程。而 Cooperative Groups 提供了 `coalesced_threads()` 函数来创建一个 coalesced threads group。

```c++
auto block = this_thread_block();

if (block.thread_rank() % 2) {
    coalesced_group active = coalesced_threads();
    ...
    active.sync();
}
```

很显然，最大的 coalesced threads group 就是一整个 warp。

## 针对 warp 的优化

### 对齐到 warp 大小

我们可以把 group size 写到模板参数里，使用静态的 group 定义，这样 thread 的大小就在编译时已知了：

```c++
thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
thread_block_tile<4>  tile4  = tiled_partition<4> (this_thread_block());
```

虽然我们可以随便定 group 的大小，但是当我们把 size 定到 warp 大小时，编译器会把同步做到 warp level，效率更高。

### 使用 warp level 指令

同时我们可以使用下面的 warp level 指令来提速：

```c++
.shfl()
.shfl_down()
.shfl_up()
.shfl_xor()
.any()
.all()
.ballot()
.match_any()
.match_all()
```
