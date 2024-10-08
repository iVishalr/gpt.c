# Achieving high performance

This sections shows some tricks used to achieve higher performance in gpt.c.

## Structure Packing

Structure Packing & Padding is used to reduce the memory footprint to store the structure in memory. Members of the structure are stored consecutively in memory. Members of the structure should be carefully placed such that there are no gaps introduced between two structure members. For example, If the structure has three members, an integer, a double and an integer, placing them one after another is not ideal. Based on the system, an integer takes 4 bytes and a double occupies 8 bytes of memory. Assume that the memory blocks in your system are 8 bytes wide. Then storing the three members will be as follows:

```
x x x x . . . . (integer - 4 bytes)
d d d d d d d d (double - 8 bytes)
i i i i . . . . (integer - 4 bytes) 
```

This is not ideal as there's considerable amount of memory being wasted between structure members due to incorrect packing. A more efficient ordering would be to place the two integers together followed by the double. And it will look as follows:

```
x x x x i i i i (2 integers - 8 bytes)
d d d d d d d d (double - 8 bytes)
```

This optimization will be in significant when using small structures, but inefficient structure packing will quickly start adding overheads as the structure becomes larger. As a consequence, many operations that depend on this structure will spend more cycles to obtain the values from structure.

The following is the tensor structure in gpt.c:

```C
typedef struct tensor {
    float *t;       // 8 bytes
    int ndims;      // 4 bytes
    int length;     // 4 bytes
    int shape[8];   // 32 bytes
} __attribute__((aligned(64))) tensor_t;
```

Structure Padding adds a certain number of empty bytes within a structure so that the data members are naturally aligned in memory. This can either be performed manually or let the compiler handle it for you.

## Memory Aligned Allocations

Memory allocations done using `malloc()` are not aligned in memory. This causes the memory to be spread across multiple cache lines or memory blocks, thus requiring more CPU cycles for retrieval. To solve this, POSIX based systems provides `aligned_malloc()`. This makes it easier to allocate memory for tensors that are aligned at the given memory alignment.

Aligned memory allocation also has another major benefit. Many of the compiler optimizations like AVX, SSE require inputs that are aligned. Without memory alignment, compiler will not enable these optimizations even though the operation seems to be suitable for vector instructions. 

The biggest beneficiary of memory alignment is cBLAS. The BLAS kernels operate much faster when the inputs are aligned. In gpt.c, there was a substantial improvement when memory alignment was enabled. The time taken per training step reduced from 2.8s to 2.1s (~25% improvement). Achieving this much optimization with just a single line change was rewarding.

Usage,

Without Memory Alignment,

```C
#include <stdlib.h>

float *a = (float *)malloc(sizeof(float) * 100);

free(a);
```

With Memory Alignment,

```C
#include <stdlib.h>

float *a = (float *)aligned_malloc(64, sizeof(float) * 100);

free(a);
```

Caveats,

On windows, to work with aligned memory, we need to use `_aligned_malloc()` and `_aligned_free()` for allocating and freeing memory respectively. These functions are accessible by including `<malloc.h>`.

## Compiler Optimizations

Enabling the optimization flags like `-O3 -Ofast -march=native` gives free performance. Behind the scenes, the compiler is doing a lot of work by analysing your code and replacing sections of your code with very fast vector instructions that can leverage SIMD. 