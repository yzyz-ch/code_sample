#include <iostream>
#include <cmath>

//cpu gemm
template <typename computeType, typename scaleType, typename inputType, typename resultType>
void cpuGEMM(const inputType* inputA,
             const inputType* inputB,
             resultType*      resultC,
             int              M,
             int              N,
             int              K,
             int              strideA,
             int              strideB,
             int              strideC,
             int              batchCount,
             scaleType        alpha,
             scaleType        beta,
             bool             transA,
             bool             transB,
             scaleType*       bias        = nullptr,
             bool             isColMajorC = true)
{
    for (int batch = 0; batch < batchCount; batch++)
    {
        const inputType* A = inputA + batch * strideA;
        const inputType* B = inputB + batch * strideB;
        resultType*      C = resultC + batch * strideC;
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                computeType sum = 0;
                for (int k = 0; k < K; k++)
                {
                    inputType a = transA ? A[k * M + m] : A[m * K + k];
                    inputType b = transB ? B[n * K + k] : B[k * N + n];
                    sum += static_cast<computeType>(a) * static_cast<computeType>(b);
                }
                unsigned ci = isColMajorC ? n * M + m : m * N + n;
                C[ci] = __float2half(alpha * static_cast<computeType>(sum) + beta * static_cast<computeType>(C[ci]));
            }
        }
    }
}