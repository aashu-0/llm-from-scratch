import torch
import torch.nn as nn
from Transformer import TransformerBlock, LayerNorm

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


# modified generate function
def generate(model,idx, max_new_tokens, context_size, temp=0.0,
             top_k = None, eos_id = None):
  for _ in range(max_new_tokens):
        # crops the current context(initial tokens) to fit model's max context size
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)  # shape (batch, n_token, vocab_size)

        logits = logits[:, -1, :]  # to extracts the last vector, shape -> (batch, vocab_size)

        if top_k is not None:
          topk_logits, _= torch.topk(logits, top_k)
          logits = torch.where(condition=logits < topk_logits[:,-1],
                               input=torch.tensor(-float('inf')).to(logits.device),
                               other=logits)

        if temp >0.0:
          logits = logits/temp
          probs = F.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)
        else:
          idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next == eos_id:
          break

        idx = torch.cat((idx, idx_next), dim=1)
  return idx
