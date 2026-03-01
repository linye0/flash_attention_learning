# Flash-attention 实现记录

## 朴素实现

Attention公式: 

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V \qquad (1)
$$

其中 $Q, K, V$ 的 shape 都是 $(N, d)$，这个计算过程涉及到两个中间矩阵，分别是：

$$
S = \frac{QK^T}{\sqrt{d}} \qquad (2)
$$

$$
P = softmax(S) \qquad (3)
$$

在实际的操作当中，因为 softmax 涉及的指数乘法很容易产生溢出，所以必须采用 safe-softmax 算法（下面提到的 softmax 默认都是 safe-softmax），即：

$$
m_i = \max_{j = 1}^{i} (x_j) , l_i = \sum_{j = 1}^{i} e^{x_j - m_N} 
$$

$$
softmax({x_i}) = \frac{e^{x_i - m_N}}{l_N} \qquad (4)
$$

因此最简单的实现就是按照 attention 的公式一步一步来：

```cpp
__global__ void softmax_v0_kernel(float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    size_t row_offset = (size_t)row * N; 
    float maxval = -CUDART_INF_F;
    float sum = 0.0;

    for (int i = 0; i < N; ++i) {
        maxval = max(maxval, S[row_offset + i]);
    }
    for (int i = 0; i < N; ++i) {
        sum += exp(S[row_offset + i] - maxval);
    }
    for (int i = 0; i < N; ++i) {
        P[row_offset + i] = exp(S[row_offset + i] - maxval) / sum;
    }
}

void launch_v0_cublas(cublasHandle_t handle, const float* Q, const float* K, const float* V, float* O, float* S, float* P, int N, int d) {
    float alpha = 1.0f / sqrtf((float)d);
    float beta = 0.0f;

    // 1. S = (1/sqrt(d)) * Q * K^T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, d, &alpha, K, d, Q, d, &beta, S, N);

    // 2. P = Softmax(S)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_v0_kernel<<<blocks, threads>>>(S, P, N);
    
    // 3. O = P * V
    float alpha_1 = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N, &alpha_1, V, d, P, N, &beta, O, d);
}
```

因为是最朴素的实现，只是为了检验思路正确性，所以矩阵乘法直接调用 cublas 库内的实现，`cublasSgemm` 的参数列表是 *(handle, 第一个矩阵是否转置, 第二个矩阵是否转置, 内积维度, 维度A, 维度B, alpha, 矩阵A, 矩阵A领先维度(实际上就是行长度), 矩阵B, 矩阵B领先长度, beta, 输出矩阵, 输出矩阵领先长度)* 。

这里要注意 `cublasSgemm` 内部默认传入的矩阵是列主序的，也就是说当我们向其传入或者传出矩阵的时候，这个矩阵都会被自动转置一次，为了适应这种逻辑，我们需要对 (1) 式子进行一些变形：

$$
S = \frac{QK^T}{\sqrt{d}} \rightarrow S^T = \frac{KQ^T}{\sqrt{d}} \qquad (5)
$$

所以第一个矩阵是转置后的 $K$（我们指定一次转置，然后传入又会有一次转置，$K^{TT}=K$），第二个矩阵是 $Q$，第三个矩阵是 $S$。

其它部分就是按照公式一步一步来，这边就已经可以看出朴素实现的低效之处了：首先是 $P$ 和 $S$ 的 shape 都是 $(N, N)$，因此会带来 $\mathcal{O}(N^2)$ 的空间复杂度；其次是 softmax 公式要求行的最大值，因为必须有一次单独的遍历，目的只是找出每行的最大值。

## online-softmax

上面所说的两个问题的根源实际上是一样的，就是我们在计算的时候必须预先要先得知 $P$ 矩阵的所有信息，于是很自然地，我们考虑是否有一种方法可以实现边读取边计算 (online) 呢？下面介绍的 online-softmax 就是为了解决这个问题：

$$
\begin{aligned}
\text{令 } l'_i &= \sum_{j = 1}^{i} e^{x_j - m_i} \\
\text{可得 } l'_i &= \sum_{j = 1}^{i} e^{x_j - m_i} \\
                &= \sum_{j = 1}^{i-1} e^{x_j - m_i} + e^{x_i - m_i}\\
                &= e^{m_{i-1} - m_i} \cdot \sum_{j= 1}^{i-1} e^{x_j - m_{i - 1}} + e^{x_i - m_i} \\
                &= e^{m_{i-1} - m_i} \cdot l'_{i-1} + e^{x_i - m_i}
\end{aligned}
$$

可以看到 $l'_i$ 的递归形式仅仅依赖 $l'_{i-1}, m_{i-1}$ 和 $m_i$，因此可以在一个循环内同时计算 $m_i$ 和 $l'_i$。

online-softmax 的步骤如下:

$$
\begin{aligned}
&\textbf{for } i \leftarrow 1, N \textbf{ do} \\
&\left| \begin{aligned}
    &\quad m_i \leftarrow \max(m_{i-1}, x_i) \\
    &\quad l'_i \leftarrow l'_{i-1} e^{m_{i-1}-m_i} + e^{x_i-m_i}
\end{aligned} \right. \\
&\textbf{end} \\
&\textbf{for } i \leftarrow 1, N \textbf{ do} \\
&\left| \begin{aligned}
    &\quad a_i \leftarrow \frac{e^{x_i - m_N}}{l'_N}
\end{aligned} \right. \\
&\textbf{end}
\end{aligned}
$$

这个算法解决了需要额外进行一次遍历进行最大值的计算的问题，实现了 $l$ 和 $m$ 的并行。
但是计算 $a$ 仍然需要一次额外的循环，我们接下来的问题是，如何消除这个循环，实现真正的 online。

单从这个式子比较难下手，但是如果我们不是一个一个计算 $a_i$，而是考虑一整段区间内的性质，会得到不同的视角。

## 从 online-softmax 到 flash-attention

我们用上面得到的新公式重新解释 attention 公式 $Attention(Q, K, V) = softmax(QK^T)V$。下面把 $Attention(Q, K, V)$ 用 $O$ 简写，针对 $O$ 当中的第 $k$ 行，计算过程如下:

$$
\begin{aligned}
&\textbf{for } i \leftarrow 1, N \textbf{ do} \\
&\left| \begin{aligned}
    &\quad x_i \leftarrow Q[k, :] * K^T[:, i] \\
    &\quad m_i \leftarrow \max(m_{i-1}, x_i) \\
    &\quad l'_i \leftarrow l'_{i-1} e^{m_{i -1} - m_i} + e^{x_i - m_i}
\end{aligned} \right. \\
&\textbf{end} \\
&\textbf{for } i \leftarrow 1, N \textbf{ do} \\
&\left| \begin{aligned}
    &\quad a_i \leftarrow \frac{e^{x_i - m_N}}{l'_N} \\
    &\quad o_i \leftarrow o_{i - 1} + a_i * V[i, :] \hspace{2em} \textbf{(这是对 O 第 k 行的迭代过程)}
\end{aligned} \right. \\
&\textbf{end} \\
&\textbf{O}[k,:] \leftarrow o_N
\end{aligned}
$$

我们对计算 $O_i$ 的式子进行变形，代入 $a_i = \frac{e^{x_i - m_N}}{l'_N}$，可得 $O_i = \sum_{j = 1}^{i} \left(\frac{e^{x_j - m_N}}{l'_N} V[j,:]\right)$。

同样的，这个式子依赖 $m_N$ 和 $l'_N$，这两个变量都需要循环结束才能得到值，我们仿照之前的方法创建一个序列 $o'$:

$$
\begin{aligned}
o'_i &= \sum_{j=1}^{i} \left( \frac{e^{x_j - m_i}}{l'_i} V[j, :] \right) \\
&= \sum_{j=1}^{i-1} \frac{e^{x_j - m_i}}{l'_i} V[j, :]  + \frac{e^{x_i - m_i}}{l'_i} V[i, :] \\
&= \frac{l'_{i-1}}{l'_i} \cdot e^{m_{i-1} - m_i} \cdot \sum_{j=1}^{i-1} \frac{e^{x_j - m_{i-1}}}{l'_{i-1}} \cdot V[j, :]  + \frac{e^{x_i - m_i}}{l'_i} V[i, :] \\
&= o'_{i-1} \frac{l'_{i-1}}{l'_i} e^{m_{i-1} - m_i} + \frac{e^{x_i - m_i}}{l'_i} V[i, :]
\end{aligned}
$$

综合上述公式，我们可以得到完整的 flash-attention 算法：

$$
\begin{aligned}
&\textbf{for } i \leftarrow 1, N \textbf{ do} \\
&\left| \begin{aligned}
    &x_i \leftarrow Q[k, :] * K^T[:, i] \\
    &m_i \leftarrow \max(m_{i-1}, x_i) \\
    &l'_i \leftarrow l'_{i-1} e^{m_{i-1}-m_i} + e^{x_i-m_i} \\
    &o'_i \leftarrow o'_{i-1} \frac{l'_{i-1}}{l'_i} e^{m_{i-1}-m_i} + \frac{e^{x_i-m_i}}{l'_i} V[i, :]
\end{aligned} \right. \\
&\textbf{end} \\
&\hspace{2em} O[k, :] \leftarrow o'_N
\end{aligned}
$$

状态 $o, l', m$ 和 $x$ 占用的空间很小，可以比较轻松的放入 GPU 的 shared_mem 中。由于此算法中的所有操作都是满足结合性的，因此它与 tiling 兼容，我们可以逐块计算状态。

$$
\begin{aligned}
&\textbf{New Notations} \\
&b : \text{the block size of the tile} \\
&\#tiles : \text{number of tiles in the row, } N = b * \#tiles \\
&x_i : \text{a vector storing the } Q * K^T \text{ value of i-th tile } [(i-1)b : ib] \\
&m_i^{(local)} : \text{the local maximum value inside } x_i. \\
&\textbf{Body} \\
&\textbf{for } i \leftarrow 1, \#tiles \textbf{ do} \\
&\left| \begin{aligned}
    &x_i \leftarrow Q[k, :] K^T [:, (i-1)b : ib] \\
    &m_i^{(local)} = \max_{j=1}^b (x_i[j]) \\
    &m_i \leftarrow \max(m_{i-1}, m_i^{(local)}) \\
    &l'_i \leftarrow l'_{i-1} e^{m_{i-1}-m_i} + \sum_{j=1}^b e^{x_i[j]-m_i} \\
    &o'_i \leftarrow o'_{i-1} \frac{l'_{i-1}}{l'_i} e^{m_{i-1}-m_i} + \sum_{j=1}^b \frac{e^{x_i[j]-m_i}}{l'_i} V[j + (i-1)b, :]
\end{aligned} \right. \\
&\textbf{end} \\
&\hspace{2em} O[k, :] \leftarrow o'_{N / b}
\end{aligned}
$$