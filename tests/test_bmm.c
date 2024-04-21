#include <stdio.h>
#include <stdlib.h>

#ifdef OSX_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

float random_uniform(float min, float max)
{
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

float *init_random_array(int B, int T, int C)
{
    printf("Initializing Random Array of shape (%d, %d, %d)\n", B, T, C);

    if (B == 0)
        B = 1;

    float *array = (float *)malloc(B * T * C * sizeof(float));
    int counter = B * T * C;
    int i = 0;
    while (counter > 0)
    {
        array[i] = random_uniform(0., 1.);
        i += 1;
        counter -= 1;
    }

    if (counter != 0 || i != B * T * C)
    {
        printf("Error in init_random_array.\n");
        return NULL;
    }
    return array;
}

void print_matrix(float *matrix, int B, int T, int C)
{
    if (B == 0)
        B = 1;

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int c = 0; c < C; c++)
            {
                printf("%f, ", matrix[b * (T * C) + t * (C) + c]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

void raw_matmul(const int _B, const int M, const int N, const int K,
                const float alpha, const float *A, const int lda,
                const float *B, const int ldb, const float beta, float *C, const int ldc)
{

    int batch = _B;
    if (_B == 0)
        batch = 1;

    for (int b = 0; b < batch; b++)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float val = 0.0f;
                for (int k = 0; k < K; k++)
                {
                    val += alpha * A[b * (M * lda) + i * (lda) + k] * B[k * ldb + j] + beta * C[b * (M * ldc) + i * ldc + j];
                }
                C[b * (M * ldc) + i * ldc + j] = val;
            }
        }
    }
}

void simple_matmul(const int M, const int N, const int K,
                   const float alpha, const float *A, const int lda,
                   const float *B, const int ldb, const float beta, float *C, const int ldc)
{
    if (C == NULL)
    {
        printf("Expected output array to be a float pointer but got NULL.\n");
        return;
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float val = 0.0f;
            for (int k = 0; k < K; k++)
            {
                val += alpha * A[i * lda + k] * B[k * ldb + j] + beta * C[i * ldc + j];
            }
            C[i * ldc + j] = val;
        }
    }
}

int main() {
    int b = 4;
    int t = 2;
    int in_features = 5;
    int out_features = 2;
    
    float *x, *W, *C;
    
    x = init_random_array(b, t, in_features);
    W = init_random_array(0, out_features, in_features);

    printf("Matrix Input:\n");
    print_matrix(x, b, t, in_features);

    printf("\nMatrix Weight:\n");
    print_matrix(W, 0, out_features, in_features);

    C = (float *)malloc(b * t * out_features * sizeof(float)); 

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, b * t, out_features, in_features, 1.0f, x, in_features, W, in_features, 0.0f, C, out_features
    );

    printf("\nOutput:\n");
    print_matrix(C, b, t, out_features);

    free(x);
    free(W);
    free(C);
}
