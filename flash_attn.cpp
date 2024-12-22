#include <iostream>


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

template <typename computeType, typename dataType>
void cpuAttentionMaskedSoftmax(const dataType* h_s,
                               dataType*       h_p,
                               const int*      h_mask,
                               unsigned        batch,
                               unsigned        head_num,
                               unsigned        seq_len,
                               unsigned        head_dim)
{
    for (int i = 0; i < batch * seq_len * head_num; i++)
    {
        const dataType* data_s = h_s + i * seq_len;
        dataType*       data_p = h_p + i * seq_len;

        computeType _max   = -INFINITY;
        computeType _sum   = 0.0;
        computeType _scale = 1 / std::sqrt(static_cast<computeType>(head_dim));

        for (int j = 0; j < seq_len; j++)
        {
            computeType s_val = (h_mask[i / (seq_len * head_num) * seq_len + j % seq_len] == 1)
                                    ? -INFINITY
                                    : static_cast<computeType>(data_s[j]);
            _max              = std::max(s_val * _scale, _max);
        }

        for (int j = 0; j < seq_len; j++)
        {
            _sum += (h_mask[i / (seq_len * head_num) * seq_len + j % seq_len] == 1)
                        ? static_cast<computeType>(0)
                        : std::exp(static_cast<computeType>(data_s[j]) * _scale - _max);
        }

        for (int j = 0; j < seq_len; j++)
        {
            data_p[j] = (h_mask[i / (seq_len * head_num) * seq_len + j % seq_len] == 1)
                            ? __float2half(0)
                            : __float2half(std::exp(static_cast<computeType>(data_s[j]) * _scale - _max) / _sum);
        }
    }
}

void cpuFMHA(unsigned      batch,
             unsigned      head_num,
             unsigned      seq_len,
             unsigned      head_dim,
             const __fp16* h_q,
             const __fp16* h_k,
             const int*    h_mask,
             const __fp16* h_v,
             __fp16*       h_o_ref)
{
    __fp16* h_s = new __fp16[batch * head_num * seq_len * seq_len]{};
    __fp16* h_p = new __fp16[batch * head_num * seq_len * seq_len]{};
    // q @ k
    {
        unsigned M        = seq_len;
        unsigned N        = seq_len;
        unsigned K        = head_dim;
        unsigned stride_q = M * K;
        unsigned stride_k = K * N;
        unsigned stride_s = M * N;
        cpuGEMM<float, float, __fp16, __fp16>(h_q,
                                              h_k,
                                              h_s,
                                              M,
                                              N,
                                              K,
                                              stride_q,
                                              stride_k,
                                              stride_s,
                                              batch * head_num,
                                              static_cast<float>(1),
                                              static_cast<float>(0),
                                              false,
                                              true,
                                              nullptr,
                                              false);
    }

    // softmax & mask
    cpuAttentionMaskedSoftmax<float, __fp16>(h_s, h_p, h_mask, batch, head_num, seq_len, head_dim);

    // p @ v
    {
        unsigned M        = seq_len;
        unsigned N        = head_dim;
        unsigned K        = seq_len;
        unsigned stride_p = M * K;
        unsigned stride_v = K * N;
        unsigned stride_o = M * N;
        cpuGEMM<float, float, __fp16, __fp16>(h_p,  // h_p
                                              h_v,
                                              h_o_ref,
                                              M,
                                              N,
                                              K,
                                              stride_p,
                                              stride_v,
                                              stride_o,
                                              batch * head_num,
                                              static_cast<float>(1),
                                              static_cast<float>(0),
                                              false,
                                              false,
                                              nullptr,
                                              false);
    }

    delete[] h_s;
    delete[] h_p;
}
