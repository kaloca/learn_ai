import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


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


class MyTransformer(nn.Module):
    def __init__(self, T=256, d_total=256, n_heads=8):
        super().__init__()
        self.T = T
        self.d_total = d_total
        self.n_heads = n_heads
        self.head_dim = d_total // n_heads
        self.W_Q = nn.Linear(d_total, d_total, bias=False)
        self.W_K = nn.Linear(d_total, d_total, bias=False)
        self.W_V = nn.Linear(d_total, d_total, bias=False)

        self.W_O = nn.Linear(d_total, d_total, bias=False)

        self.L1 = nn.Linear(d_total, 4 * d_total)
        self.L2 = nn.Linear(4 * d_total, d_total)
        self.Lnorm1 = nn.LayerNorm(d_total)
        self.Lnorm2 = nn.LayerNorm(d_total)
        self.gelu = nn.GELU()

    def multi_head_attention(self, X):
        T = X.shape[0]
        Q = self.W_Q(X).view(T, self.n_heads, self.head_dim)
        K = self.W_K(X).view(T, self.n_heads, self.head_dim)
        V = self.W_V(X).view(T, self.n_heads, self.head_dim)
        final = []
        for i in range(self.n_heads):
            attn = single_head_attention(Q[:, i, :], K[:, i, :], V[:, i, :])
            final.append(attn)
        return self.W_O(torch.cat(final, dim=-1))

    def forward(self, x):
        x = self.Lnorm1(x)
        x = x + self.multi_head_attention(x)
        x = self.Lnorm2(x)
        x = x + self.L2(self.gelu(self.L1(x)))

        return x


class MyGPT(nn.Module):
    def __init__(self, vocab_size=2000, d_total=256, n_blocks=6, T=2000):
        super().__init__()

        self.T = T
        self.vocab_size = vocab_size
        self.d_total = d_total
        self.n_blocks = n_blocks

        self.token_emb = nn.Embedding(vocab_size, d_total)
        self.pos_emb = nn.Embedding(T, d_total)

        self.transformer_blocks = nn.ModuleList(
            [MyTransformer() for i in range(n_blocks)]
        )
        self.lnorm = nn.LayerNorm(d_total)
        self.linear = nn.Linear(d_total, vocab_size)

    def forward(self, idx):
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(self.T).to("cuda"))
        x = tok + pos

        for i in range(self.n_blocks):
            x = self.transformer_blocks[i].forward(x)

        x = self.lnorm(x)
        x = self.linear(x)
        return x

    def generate(self, prompt, max_new_tokens=256):
        # output = []
        prompt_tensor = torch.tensor([0] * (self.T - len(prompt)) + prompt).to("cuda")
        for i in range(max_new_tokens):
            fwd = self.forward(prompt_tensor)
            last_c = fwd[-1].argmax()

            prompt_tensor = torch.cat(
                [prompt_tensor[1:], torch.tensor([last_c]).to("cuda")]
            )
            # print(last_c)
        return prompt_tensor[self.T - max_new_tokens :]


class TrainGPT:
    def __init__(self):
        self.vocab = {}

    def parse_book(self, path):
        self.text = open(path).read()

    def build_vocab(self):
        cur = 0
        for char in self.text:
            if char not in self.vocab.keys():
                self.vocab[char] = cur
                cur = cur + 1

        self.vocab_size = len(self.vocab.keys())
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize_book(self):
        self.encoded_text = [self.vocab[c] for c in self.text]

    def train(self, n_steps):
        for i in tqdm(range(n_steps)):
            i = torch.randint(0, len(self.encoded_text) - self.T, (1,)).item()
            chunk = self.encoded_text[i : i + self.T + 1]

            input = torch.tensor(chunk[:-1]).to("cuda")
            target = torch.tensor(chunk[1:]).to("cuda")

            logits = self.model.forward(input)

            loss = F.cross_entropy(logits.view(-1, self.vocab_size), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def init_model(self, T=1024):
        self.T = T
        self.model = MyGPT(T=T, vocab_size=self.vocab_size).to("cuda")
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def prompt(self, prompt):
        encoded_prompt = [self.vocab[c] for c in prompt]
        generation = self.model.generate(encoded_prompt)
        decoded_prompt = "".join([self.inv_vocab[k.item()] for k in generation])

        return decoded_prompt


gpt = TrainGPT()

gpt.parse_book("foundation.txt")
gpt.build_vocab()
gpt.tokenize_book()
gpt.init_model(T=256)
gpt.train(n_steps=10000)
response = gpt.prompt(
    "testing this shit out. my name is gabriel. i am from. i wanted to know what happens if someone from"
)
print(response)
