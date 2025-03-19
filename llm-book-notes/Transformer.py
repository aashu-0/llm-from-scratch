from MHA import MultiHeadAttention
import torch
import torch.nn as nn


# layer norm
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5

        # two learnable parameter 'scale' and 'shift'
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, keepdim= True, unbiased= False) #biased variance

        norm_x = (x - mean) / torch.sqrt(var+self.eps)
        
        return self.scale*norm_x + self.shift



# GELU Activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gelu = 0.5 * x * (1 + torch.tanh(
                          torch.sqrt(torch.tensor(2.0/torch.pi)) *
                          (x+0.044715 * torch.pow(x,3))
        ))
        return gelu



# feed forward network with gelu activations
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], 4*config['emb_dim']),
            GELU(),
            nn.Linear(4*config['emb_dim'], config['emb_dim'])
        )

    def forward(self, x):
        return self.layers(x)



#Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_in= config['emb_dim'],
            d_out= config['emb_dim'],
            context_length= config['context_length'],
            num_heads = config['n_heads'],
            dropout=config['drop_rate'],
            qkv_bias=config['qkv_bias']
        )
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config['emb_dim'])
        self.norm2 = LayerNorm(config['emb_dim'])
        self.drop_shortcut = nn.Dropout(config['drop_rate'])

    def forward(self, x):

        # attn block
        shortcut = x
        x = self.norm1(x)   # pre-layer norm (opposite to original transformer model)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        #ffn block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x +shortcut
        return x