import math
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
        (Q @ K.T)
        / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32, device=K.device)),
        dim=-1,
    )

    return score @ V


class MyTransformer(nn.Module):
    def __init__(self, T=256, d_total=256, n_heads=8, dropout=0.1):
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

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(self.T, self.T, dtype=torch.bool))
        )

    def old_multi_head_attention(self, X):
        T = X.shape[0]
        Q = self.W_Q(X).view(T, self.n_heads, self.head_dim)
        K = self.W_K(X).view(T, self.n_heads, self.head_dim)
        V = self.W_V(X).view(T, self.n_heads, self.head_dim)
        final = []
        for i in range(self.n_heads):
            attn = single_head_attention(Q[:, i, :], K[:, i, :], V[:, i, :])
            final.append(attn)
        return self.W_O(torch.cat(final, dim=-1))

    # fast
    def multi_head_attention(self, X):
        B, T, _ = X.shape
        Q = (
            self.W_Q(X).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B,H,T,D)
        K = (
            self.W_K(X).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B,H,T,D)
        V = (
            self.W_V(X).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B,H,T,D)

        scores = Q @ K.transpose(-2, -1)  # (B,H,T,T)
        scores = scores / (self.head_dim**0.5)

        mask = self.causal_mask[:T, :T]  # (T,T)
        scores = scores.masked_fill(~mask[None, None, :, :], -float("inf"))

        probs = torch.softmax(scores, dim=-1)  # (B,H,T,T)
        probs = self.attn_dropout(probs)

        out = probs @ V  # (B,H,T,D)

        out = out.transpose(1, 2).reshape(B, T, self.d_total)  # (B,T,C)
        out = self.W_O(out)
        out = self.resid_dropout(out)
        return out

    def forward(self, x):
        attn_in = self.Lnorm1(x)
        attn_out = self.multi_head_attention(attn_in)
        x = x + attn_out

        mlp_in = self.Lnorm2(x)
        h = self.L1(mlp_in)
        h = self.gelu(h)
        mlp_out = self.L2(h)
        mlp_out = self.resid_dropout(mlp_out)
        x = x + mlp_out

        return x


class MyGPT(nn.Module):
    def __init__(
        self, vocab_size=2000, d_total=256, n_blocks=6, T=2000, n_heads=8, dropout=0.1
    ):
        super().__init__()

        self.T = T
        self.vocab_size = vocab_size
        self.d_total = d_total
        self.n_blocks = n_blocks

        self.token_emb = nn.Embedding(vocab_size, d_total)
        self.pos_emb = nn.Embedding(T, d_total)

        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [
                MyTransformer(T=T, d_total=d_total, n_heads=n_heads)
                for _ in range(n_blocks)
            ]
        )
        self.lnorm = nn.LayerNorm(d_total)
        self.linear = nn.Linear(d_total, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device)).unsqueeze(0)
        x = self.emb_dropout(tok + pos)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.lnorm(x)
        x = self.linear(x)
        return x

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=256, temperature=1.0, top_k=None):
        device = next(self.parameters()).device
        prompt_tensor = torch.tensor(prompt, device=device)
        for i in range(max_new_tokens):
            inp = prompt_tensor[-self.T :].unsqueeze(0)
            logits = self.forward(inp).squeeze(0)[-1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            last_c = torch.multinomial(probs, num_samples=1)
            prompt_tensor = torch.cat([prompt_tensor, last_c])
        return prompt_tensor[-max_new_tokens:]


B = 32


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
        self.encoded_text = torch.tensor(
            [self.vocab[c] for c in self.text], dtype=torch.long
        ).to("cuda")

    def train_old(self, n_steps):
        for i in tqdm(range(n_steps)):
            i = torch.randint(0, len(self.encoded_text) - self.T, (1,)).item()
            chunk = self.encoded_text[i : i + self.T + 1]

            input = chunk[:-1]
            target = chunk[1:]

            logits = self.model.forward(input)

            loss = F.cross_entropy(logits.view(-1, self.vocab_size), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        self.model.train()
        for i in tqdm(range(self.n_steps)):
            ix = torch.randint(0, len(self.encoded_text) - self.T, (B,))
            input = torch.stack(
                [self.encoded_text[i : i + self.T] for i in ix]
            )  # (B, T)
            target = torch.stack(
                [self.encoded_text[i + 1 : i + self.T + 1] for i in ix]
            )  # (B, T)

            logits = self.model.forward(input)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            if i % 5000 == 0:
                tqdm.write(f"step {i}, loss: {loss.item():.4f}")
                tqdm.write(
                    self.prompt(
                        "testing this shit out. my name is gabriel. i am from. i wanted to know what happens if someone from"
                    )
                )

    def init_model(self, T=1024, n_steps=50000, warmup_steps=500):
        self.T = T
        self.n_steps = n_steps
        self.model = MyGPT(T=T, vocab_size=self.vocab_size).to("cuda")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (n_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def prompt(self, prompt):
        was_training = self.model.training
        self.model.eval()
        encoded_prompt = [self.vocab[c] for c in prompt]
        generation = self.model.generate(encoded_prompt)
        decoded_prompt = "".join([self.inv_vocab[k.item()] for k in generation])

        if was_training:
            self.model.train()

        return decoded_prompt

    def save(self, path="model.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pt"):
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    gpt = TrainGPT()

    gpt.parse_book("foundation.txt")
    gpt.build_vocab()
    gpt.tokenize_book()
    gpt.init_model(T=256, n_steps=50000)
    gpt.train()
    gpt.save()

    response = gpt.prompt(
        "testing this shit out. my name is gabriel. i am from. i wanted to know what happens if someone from"
    )
    print(response)
