pair = lambda x: x if isinstance(x, tuple) else (x, x)
from functools import partial
import torch
from torch import nn 


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLPMixer(nn.Module):
    def __init__(self, depth,feaure_in,  dim, num_patches, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0., batched=True):
        super(MLPMixer, self).__init__()
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.tokenizer = nn.Linear(feaure_in,  dim)
        self.mixer = nn.Sequential(*[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, self.chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, self.chan_last))
        ) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.predictor = nn.Linear(dim, 1)
        self.batched = batched

    def forward(self, x):
        x = x.to(self.device)
        x = self.tokenizer(x)
        x = self.mixer(x)
        x = x.mean(dim=1) if self.batched else  x.mean(dim=0)
        y = self.predictor(x)
        return y