#include <cmath>
#include <vector>
#include <omp.h>
#include <algorithm>

/**
 * @brief CPU 局部抽样参考算法
 * @param num_rows 只计算前多少行进行校验（建议 64 或 100）
 */
void cpu_attention_reference_partial(
    const float* Q, const float* K, const float* V, float* O, 
    int N, int d, int num_rows) 
{
    float scale = 1.0f / sqrtf((float)d);
    
    // 限制计算行数，防止 N 太大时 CPU 跑不动
    int actual_rows = std::min(num_rows, N);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < actual_rows; ++i) {
        // 1. 计算 S[i, :] = (Q[i, :] @ K.T) * scale
        std::vector<float> row_S(N);
        float max_val = -1e20f;

        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int k = 0; k < d; ++k) {
                dot += Q[i * d + k] * K[j * d + k];
            }
            row_S[j] = dot * scale;
            if (row_S[j] > max_val) max_val = row_S[j];
        }

        // 2. 计算 Softmax: P[i, :] = exp(S[i, :] - max_val) / sum(...)
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            row_S[j] = expf(row_S[j] - max_val);
            sum_exp += row_S[j];
        }

        // 3. 计算 O[i, :] = P[i, :] @ V
        for (int k = 0; k < d; ++k) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                weighted_sum += (row_S[j] / sum_exp) * V[j * d + k];
            }
            O[i * d + k] = weighted_sum;
        }
    }
}