# Notes on Matrix Multiplication

Matrix Multiplication is the most expensive operation in any neural network. Operations other than matrix multiplication are performant right out of the box with compiler optimizations. To accelerate matrix multiplication on CPU, gpt.c uses a third party library called OpenBLAS. OpenBLAS provides a C interface for many of the BLAS routines. These BLAS routines are highly optimized algorithms that operate on matrices and accelerate computation by making use of caches and other clever techniques such as blocking and matrix packing.

Since gpt.c uses OpenBLAS to accelerate matrix multiplication, it's worth learning about the cblas function for matrix multiplication.

## cBLAS SGEMM

The `cblas_sgemm` function is repsonsible for performing matrix multiplication. This function mainly operates on single precision floats. The following is the function declaration of `cblas_sgemm`.

```C
void cblas_sgemm(
    OPENBLAS_CONST enum CBLAS_ORDER Order, 
    OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, 
    OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
    OPENBLAS_CONST blasint M, 
    OPENBLAS_CONST blasint N, 
    OPENBLAS_CONST blasint K, 
    OPENBLAS_CONST float alpha, 
    OPENBLAS_CONST float *A, 
    OPENBLAS_CONST blasint lda, 
    OPENBLAS_CONST float *B, 
    OPENBLAS_CONST blasint ldb, 
    OPENBLAS_CONST float beta, 
    float *C, 
    OPENBLAS_CONST blasint ldc
);
```

**Operation performed:** `C ← αAB + βC`

`Order` - Whether matrix is stored in RowMajor or ColumnMajor Order. C uses row major order by default. All cblas_sgemm functions in gpt.c use `CblasRowMajor` for this argument.

`TransA` - Whether matrix A is supposed to be transposed before matrix mulitplication. No Transpose - `CblasNoTrans` or Transpose - `CblasTrans`.

`TransB` - Whether matrix B is supposed to be transposed before matrix mulitplication. No Transpose - `CblasNoTrans` or Transpose - `CblasTrans`.

`M` - Number of rows in Matrix A.

`N` - Number of columns in Matrix B.

`K` - Number of columns in Matrix A or Number of rows in Matrix B. Both should match.

`alpha` - Scaling factor for Matrix A.

`A` - Float pointer to Matrix A.

`lda` - Leading dimension of Matrix A. Usually equal to number of columns in Matrix A.

`B` - Float pointer to Matrix B.

`ldb` - Leading dimension of Matrix B. Usually equal to number of columns in Matrix B.

`beta` - Scaling factor for Matrix C. Setting it to 0, will make matmul output to be stored in C. If set to 1, matmul output will be accumulated (sum) in C.

`C` - Float pointer to Matrix C.

`ldc` - Leading dimension of Matrix C. Usually equal to number of columns in Matrix C.

## Examples

To compute the attention matrix, in python

```py
B = 1
T = 1024
C = 768
nheads = 12

q = torch.randn((B, nheads, T, C // nheads)) # Shape [B, h, T, hs]
k = torch.randn((B, nheads, T, C // nheads)) # Shape [B, h, T, hs]

att = (q @ k.transpose(-2, -1)) * 1 / math.sqrt(C // nheads) # Shape [B, h, T, T] = ([B, h, T, hs] @ [B, h, hs, T]) / sqrt(hs)
```

Using cblas, the same thing can be computed as follows:

```C
for (int i = 0; i < B; i++) {
    for (int j = 0; j < nheads; j++) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            T, T, hs,
            scale , q->t + b * n_heads * T * hs + h * T * hs, hs,
            k->t + b * n_heads * T * hs + h * T * hs, hs,
            0.0f, att->t + b * n_heads * T * T + h * T * T, T
        );
    }
}
```

cblas can only perform 2D matrix multiplication. For more dimensions, cblas should be called multiple times with different views of the matrix.

`M = T` as `q` has `T` rows

`N = T` as `k` has `T` columns (after transpose)

`K = hs` as columns in `q` = `hs` and rows in `k` (after transpose) = `hs`

`lda = hs` as `q` has `hs` columns

`ldb = hs` as `k` has `hs` columns before transpose

`ldc = T` as `att` has `T` columns

`alpha = scale = 1 / sqrt(hs)`

`beta = 0` as we want to store the value in att, and not accumulate in att.

*Note: From personal experience while creating gpt.c, figuring out these parameters was the hardest thing to do. A small change in these values can result in different output values.*