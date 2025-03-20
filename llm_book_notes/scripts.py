import torch
import torch.nn as nn
import math
import tiktoken
from torch.utils.data import Dataset, DataLoader


# 1. custom dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.inputs_ids = []
        self.target_ids = []

        # tokenize the text
        token_ids = tokenizer.encode(txt)

        #uses a sliding window approach
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+1+max_length]

            self.inputs_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # return total num of rows in the dataset
    def __len__(self):
        return len(self.inputs_ids)
    
    # return a single row
    def __getitem__(self, index):
        return self.inputs_ids[index], self.target_ids[index]
    

# 2. Custom dataloader
def create_dataloders_v1(txt,
                         batch_size=4,
                         max_length= 256,
                         stride= 128,  # stride -> num of positions the input shift across batches
                         shuffle= True,
                         drop_last = True,
                         num_workers = 0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset,
                            batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=num_workers)
    
    return dataloader


#--------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()

        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # .view() -> to reshape tensors
        # split d_out into num_heads, head_dim
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose from shape (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # attn scores -> dot product of queries and keys for each head
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / math.sqrt(keys.shape[-1]), dim = -1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        # transposing makes tensor non-contiguous

        # therefore before flattening into shape (b, num_tokens, self.d_out) make into contiguous 
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # self.d_out = self.num_heads * self.head_dim

        context_vec = self.out_proj(context_vec)
        return context_vec


#--------------------------------------------------

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
    

#--------------------------------------------------

#GPT model
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = nn.Dropout(config['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config)
              for _ in range(config['n_layers'])]
        )
        self.final_norm = LayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias = False)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        tok_embs = self.tok_emb(input_ids)

        pos_embs = self.pos_emb(torch.arange(seq_len, device= input_ids.device))

        x = tok_embs + pos_embs
        x = self.drop_emb(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

# a function to generate text
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # crops the current context(initial tokens) to fit model's max context size
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)  # shape (batch, n_token, vocab_size)

        logits = logits[:, -1, :]  # to extracts the last vector, shape -> (batch, vocab_size)

        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim = True)  # shape ->(batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # shape -> (batch, n_tokens +1)

    return idx