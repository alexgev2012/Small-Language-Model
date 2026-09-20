"""A randomly initialized byte-level causal Transformer; no pretrained assets."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    context: int = 256
    width: int = 256
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if min(self.context, self.width, self.layers, self.heads) < 1:
            raise ValueError('Model dimensions must be positive.')
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError('Width must divide evenly into heads; dropout must be in [0, 1).')


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                 nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.projection(attention.transpose(1, 2).contiguous().view(batch, length, width))
        return x + self.mlp(self.norm2(x))


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(256, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, 256, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if not 1 <= tokens.shape[1] <= self.config.context:
            raise ValueError('Sequence length exceeds the model context or is empty.')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.tokens(tokens) + self.positions(positions)
        return self.output(self.norm(self.blocks(x)))

    @torch.inference_mode()
    def generate(self, prompt, count=300, temperature=0.8, top_k=40):
        if not prompt or count < 0 or temperature <= 0 or not 1 <= top_k <= 256:
            raise ValueError('Use a nonempty prompt, nonnegative length, positive temperature and top-k 1..256.')
        self.eval()
        device = next(self.parameters()).device
        result = bytearray(prompt.encode('utf-8'))
        for _ in range(count):
            x = torch.tensor([list(result[-self.config.context:])], dtype=torch.long, device=device)
            logits = self(x)[0, -1] / temperature
            values, indices = torch.topk(logits, top_k)
            choice = torch.multinomial(F.softmax(values, dim=-1), 1)
            result.append(indices[choice].item())
        # Untrained models may emit invalid UTF-8; replacement preserves visible output.
        return result.decode('utf-8', errors='replace')


def save_checkpoint(path, model, optimizer, step, data_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'format': 1, 'config': asdict(model.config), 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'step': step, 'data_hash': data_hash}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('format') != 1:
        raise ValueError('Unsupported checkpoint format.')
    model = TextModel(Config(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint
