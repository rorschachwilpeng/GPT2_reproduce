from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#----------------------------------------------------------------------------------------------------



@dataclass
class GPTConfig:
    block_zie: int = 256
    vocab_size: int =65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),#weights of token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),#weights of positional embedding
            #layers代表？
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)],#attention block里的内容？
            ln_f = nn.LayerNorm(config.n_embd),#GPT2在self attention和Linear之间的一层新加入（暂时不知道有什么用？）
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)#Linear层