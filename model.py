from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 2048
    n_embd: int = 1024
    n_layer: int = 16
    n_head: int = 16
    dropout: float = 0.1
    use_checkpointing: bool = True


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with a single fused QKV projection."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.num_heads = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, emb = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            wei = wei.masked_fill(mask == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, emb)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple MLP (SwiGLU-style) block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        ffn_hidden = int(4 * config.n_embd / 3)
        self.w1 = nn.Linear(config.n_embd, ffn_hidden)
        self.w2 = nn.Linear(config.n_embd, ffn_hidden)
        self.w3 = nn.Linear(ffn_hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w3(F.silu(self.w1(x)) * self.w2(x))
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        bsz, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(seq_len, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        if self.config.use_checkpointing and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        logits = logits.view(bsz * seq_len, -1)
        targets = targets.view(bsz * seq_len)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
