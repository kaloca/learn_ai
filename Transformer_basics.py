import numpy as np
import torch


def single_head_attention(Q, K, V):
    # Context length = T = 2000
    # d_k = 64
    # X = (2000, 64)
    # Q, K, V = (64, 64)
    score = torch.softmax(
        (Q @ K.T) / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32)), dim=-1
    )

    return score @ V


def multi_head_attention(X, d_total, n_heads, W_O):
    # d_total = embedding dim
    head_dim = d_total // n_heads
    W_Qs = [torch.randn(d_total, head_dim) for _ in range(n_heads)]
    W_Ks = [torch.randn(d_total, head_dim) for _ in range(n_heads)]
    W_Vs = [torch.randn(d_total, head_dim) for _ in range(n_heads)]

    final = []
    for i in range(n_heads):
        Q = X @ W_Qs[i]
        K = X @ W_Ks[i]
        V = X @ W_Vs[i]

        attn = single_head_attention(Q, K, V)
        final.append(attn)

    return torch.cat(final, dim=-1) @ W_O


T, d_k = 5, 4
Q = torch.randn(T, d_k)
K = torch.randn(T, d_k)
V = torch.randn(T, d_k)
out = single_head_attention(Q, K, V)
print(out.shape)

T, d_total, n_heads = 5, 8, 2
X = torch.randn(T, d_total)
W_O = torch.randn(d_total, d_total)
out = multi_head_attention(X, d_total, n_heads, W_O)
print(out.shape)
