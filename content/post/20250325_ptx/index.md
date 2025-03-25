---
title: 在 CUDA 中使用 PTX
description: 
slug: ptx
date: 2025-03-25 14:23:56+0800
# image: cover.jpg
categories:
    - CUDA
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

以下内容翻译自 [CUDA 官方的 PTX 使用说明](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)，并进行了一些整理。

## ASM 命令

我们从 ASM 指令的格式讲起，ASM 指令的格式如下：

```c++
asm("template-string" : "constraint"(output) : "constraint"(input));
```

一条简单的 ASM 语句如下所示：

```c++
asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));
```

从这个格式和示例，我们可以注意到以下两点：

- 在 `asm()` 括号内的内容分为三部分，分别为模板字符串、输出和输入，这三部分以冒号分隔，但在输入输出中的不同变量以逗号分隔。
- 输出和输入前都有一个 “约束”。

接下来我们依次讨论一下这两点的细节，先从模板字符串开始。

### 模板字符串

在模板字符串中主要需要注意的就是 “%n” 了。这里的 “n” 对应了后面操作符的序号，即 “%0” 对应第一个操作符，“%1” 对应第二个操作符，以此类推。所以顺序可以任意指定，如之前的示例

```c++
asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));
```

在概念上对应

```.asm
add.s32 i, j, k;
```

而

```c++
asm("add.s32 %0, %2, %1;" : "=r"(i) : "r"(k), "r"(j));
```

在概念上和上面的语句是相同的。同时同一个操作符也可以重复出现，如

```c++
asm("add.s32 %0, %1, %1;" : "=r"(i) : "r"(k));
```

这在概念上等同于

```.asm
add.s32 i, k, k;
```

#### 省略

当没有输入操作符时，后面的冒号部分可以省略掉，如：

```c++
asm("mov.s32 %0, 2;" : "=r"(i));
```

而当没有输出操作符时，输出部分的内容可以空置，如：

```c++
asm("mov.s32 r1, %0;" :: "r"(i));
```

#### 转义

当需要在 PTX 指令中使用 “%” 时，需要用 “%%” 进行转义：

```c++
asm("mov.u32 %0, %%clock;" : "=r"(x));
```

#### 多条语句

在一个 `asm()` 语句中可以放置多个语句。为了在 PTX 中间文件中生成可读的代码，最好在每一个指令后面用 “\n\t” 结尾，如

```c++
__device__ int cube (int x)
{
  int y;
  asm(".reg .u32 t1;\n\t"              // temp reg t1
      " mul.lo.u32 t1, %1, %1;\n\t"    // t1 = x * x
      " mul.lo.u32 %0, t1, %1;"        // y = t1 * x
      : "=r"(y) : "r" (x));
  return y;
}
```

与

```c++
__device__ int cond (int x)
{
  int y = 0;
  asm("{\n\t"
      " .reg .pred %p;\n\t"
      " setp.eq.s32 %p, %1, 34;\n\t" // x == 34?
      " @%p mov.s32 %0, 1;\n\t"      // set y to 1 if true
      "}"                            // conceptually y = (x==34)?1:y
      : "+r"(y) : "r" (x));
  return y;
}
```

### 约束

#### 寄存器约束

在上面的示例中看到的约束都有字母 “r”，这里的 “r” 指的是 32 位整数寄存器。关于寄存器的约束列表如下：

```
"h" = .u16 reg
"r" = .u32 reg
"l" = .u64 reg
"q" = .u128 reg
"f" = .f32 reg
"d" = .f64 reg
```

注意 “q” 约束只能在支持 `__int128` 的机器上使用。

#### 立即整数约束

同时还有 “n” 约束，表示已知的立即整数（immediate integer operands），如

```c++
asm("add.u32 %0, %0, %1;" : "=r"(x) : "n"(42));
```

这在概念上等于

```.asm
add.u32 r1, r1, 42;
```

#### 常字符数组约束

约束 “C” 用来表示常字符数组（array of const char），这个字符数组的内容必须是编译时已知的。这个的主要目的是在编译时改变 PTX 命令的 “modes”，例如：

```c++
constexpr int mode_rz = 0;
constexpr int mode_rn = 1;

template <int mode>
struct helper;

template<> struct helper<mode_rz> {
    static constexpr const char mode[] = ".rz";
};

template<> struct helper<mode_rn> {
    static constexpr const char mode[] = ".rn";
};

template <int rounding_mode>
__device__ float compute_add(float a, float b) {
    float result;
    asm ("add.f32%1 %0,%2,%3;" : "=f"(result)
                            : "C"(helper<rounding_mode>::mode),
                              "f"(a), "f"(b));
    return result;
}

__global__ void kern(float *result, float a, float b) {
    *result++ = compute_add<mode_rn>(a,b); // generates add.f32.rn
    *result   = compute_add<mode_rz>(a,b); // generates add.f32.rz
}
```

我们现在知道 “C” 约束后面跟的应该是一个字符数组地址，这个地址指向的变量 `V` 必须满足下面的约束：

- `V` 是 `static` 存储的；
- `V` 的类型是 array of const char；
- `V` 是用常量初始化的；
- 如果 `V` 是一个 static class 的成员，`V` 的初始化必须也在这个类中。

并且，如果 `V` 中有 '\0' 或 '0' 作为结尾，这个结尾会被去除。例如：

```c++
struct S1 {
static constexpr char buf1[] = "Jumped";
static constexpr char buf2[] = {'O', 'v', 'e', 'r', 0};
};

template <const char *p1, const char *p2, const char *p3>
__device__ void doit() {
asm volatile ("%0 %1 %2" : : "C"(p1), "C"(p2), "C"(p3));
}

struct S2 {
static const char buf[];
};
const char S2::buf[] = "this";

const char buf3[] = "Jumped";
extern const char buf4[];

__global__ void foo() {
    static const char v1[] = "The";
    static constexpr char v2[] = "Quick";
    static const char v3[] = { 'B' , 'r' , 'o', 'w', 'n', 0 };
    static constexpr char v4[] = { 'F', 'o', 'x', 0 };

    //OK: generates 'The Quick Brown Fox Jumped Over' in PTX
    asm volatile ("%0 %1 %2 %3 %4 %5" : :  "C"(v1) , "C"(v2), "C"(v3),  "C"(v4), "C"(S1::buf1), "C"(S1::buf2) );

    //OK: generates 'Brown Fox Jumped' in PTX
    doit<v3, v4, buf3>();


    //error cases

    const char n1[] = "hi";

    //error: argument to "C" constraint is not a constant expression
    asm volatile ("%0" :: "C"(n1));

    //error: S2::buf was not initialized at point of declaration
    asm volatile ("%0" :: "C"(S2::buf));

    //error: buf4 was not initialized
    asm volatile ("%0" :: "C"(buf4));
}
```

#### 8 位寄存器的约束

8 位寄存器没有特殊的字母来指定约束，但可以接收大于 8 位的类型。例如

```c++
__device__ void copy_u8(char* in, char* out) {
    int d;
    asm("ld.u8 %0, [%1];" : "=r"(d) : "l"(in) : "memory");
    *out = d;
}
```

会生成

```.asm
ld.u8 r1, [rd1];
st.u8 [rd2], r1;
```

#### 约束中的修饰符

我们注意到前面的示例中有时在 “r” 前有 “=”，这个 “=” 修饰符表示这个寄存器用于输出。同时还有一个 “+” 修饰符表示这个寄存器同时被读出和写入。例如

```c++
asm("add.s32 %0, %0, %1;" : "+r"(i) : "r" (j));
```

这里需要加这个约束的主要原因是，在实际执行时还需要有加载和存储字符串的过程。如对于这个 PTX 语句

```c++
asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));
```

实际编译器的输出是

```.asm
ld.s32 r1, [j];
ld.s32 r2, [k];
add.s32 r3, r1, r2;
st.s32 [i], r3;
```

所以约束在这里变得很重要，约束中作为输入的操作符会在实际语句执行前被加载进寄存器，而结果会在实际语句后被写入寄存器。

## 使用 ASM 命令的一些问题

下面讨论一下使用 ASM 命令可能会遇到的一些问题。

### 命名空间冲突

对于上面我们举例过的 `cube` 函数：

```c++
__device__ int cube (int x)
{
  int y;
  asm(".reg .u32 t1;\n\t"              // temp reg t1
      " mul.lo.u32 t1, %1, %1;\n\t"    // t1 = x * x
      " mul.lo.u32 %0, t1, %1;"        // y = t1 * x
      : "=r"(y) : "r" (x));
  return y;
}
```

如果我们把它写成 `inline` 并在代码中多次使用，我们会受到一个报错，说临时寄存器 `t1` 重复声明。这时我们可以：

- 不要 `inline` 这个函数
- 把使用 `t1` 的部分用大括号包起来，如：

```c++
__device__ int cube (int x)
{
  int y;
  asm("{\n\t"                        // use braces for local scope
      " reg .u32 t1;\n\t"            // temp reg t1,
      " mul.lo.u32 t1, %1, %1;\n\t"  // t1 = x * x
      " mul.lo.u32 %0, t1, %1;\n\t"  // y = t1 * x
      "}"
      : "=r"(y) : "r" (x));
  return y;
}
```

### 内存空间冲突

`asm()` 语句无法得知传进来的寄存器在哪个内存空间里，所以用户需要确定使用了合适的 PTX 指令。在 `sm_20` 和以上版本中，所有传给 `asm()` 的指针都是 generic address。

### 不正确的优化

一般直接使用 `asm()`，编译器会认为这句话唯一的作用是改变了输出变量，不会有其它作用，所以有时编译器会对这些语句进行优化。而使用 `asm volatile()` 会确保这个语句在生成 PTX 时不会被删除或移动顺序。如

```.asm
asm volatile ("mov.u32 %0, %%clock;" : "=r"(x));
```

此外，内存操作也是类似。一般认为被写入的内存都会被放在输出操作符的位置上。但是如果有隐藏的内存读写（比如间接访存时的地址计算），或者我们就是想在生成 PTX 时去掉 `asm()` 语句附近的内存优化，我们可以在语句中增添第三个冒号，并加上 “memory”。例如

```c++
asm volatile ("mov.u32 %0, %%clock;" : "=r"(x) :: "memory");
asm ("st.u32 [%0], %1;" :: "l"(p), "r"(x) : "memory");
```

### 不正确的 PTX

由于编译器是不会检查 `asm()` 语句的内部的，所以错误只会在 `ptxas` 中显示。
