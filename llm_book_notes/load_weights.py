import numpy as np
import torch

# utility function to check two tensors or array are of same dim
def assign(left, right):
  if left.shape != right.shape:
    raise ValueError(f'Shape mismatch: {left.shape} != {right.shape}')

  return torch.nn.Parameter(torch.tensor(right))


# manually matching weights of pretrained gpt2 to out GPTModel Function
def load_weights_into_gpt(gpt, params):
  gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
  gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

  for b in range(len(params['blocks'])):

    # query, key, value weight
    q_w, k_w, v_w = np.split(
        (params['blocks'][b]['attn']['c_attn'])['w'],
        3, axis=-1)
    gpt.trf_blocks[b].attn.W_query.weight = assign(gpt.trf_blocks[b].attn.W_query.weight, q_w.T)
    gpt.trf_blocks[b].attn.W_key.weight = assign(gpt.trf_blocks[b].attn.W_key.weight, k_w.T)
    gpt.trf_blocks[b].attn.W_value.weight = assign(gpt.trf_blocks[b].attn.W_value.weight, v_w.T)

    # query, key, value bias
    q_b, k_b, v_b = np.split(
        (params['blocks'][b]['attn']['c_attn'])['b'],
        3, axis=-1)
    gpt.trf_blocks[b].attn.W_query.bias = assign(gpt.trf_blocks[b].attn.W_query.bias, q_b)
    gpt.trf_blocks[b].attn.W_key.bias = assign(gpt.trf_blocks[b].attn.W_key.bias, k_b)
    gpt.trf_blocks[b].attn.W_value.bias = assign(gpt.trf_blocks[b].attn.W_value.bias, v_b)

    # out_proj weight and bias
    gpt.trf_blocks[b].attn.out_proj.weight = assign(gpt.trf_blocks[b].attn.out_proj.weight, params['blocks'][b]['attn']['c_proj']['w'].T)
    gpt.trf_blocks[b].attn.out_proj.bias = assign(gpt.trf_blocks[b].attn.out_proj.bias, params['blocks'][b]['attn']['c_proj']['b'])

    # feedforward layer
    gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params['blocks'][b]['mlp']['c_fc']['w'].T)
    gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params['blocks'][b]['mlp']['c_fc']['b'])

    gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params['blocks'][b]['mlp']['c_proj']['w'].T)
    gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params['blocks'][b]['mlp']['c_proj']['b'])

    # layer norm1
    gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params['blocks'][b]['ln_1']['g'])
    gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params['blocks'][b]['ln_1']['b'])

    #layer norm2
    gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params['blocks'][b]['ln_2']['g'])
    gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params['blocks'][b]['ln_2']['b'])

  #layer norm3
  gpt.final_norm.scale = assign(gpt.final_norm.scale, params['g'])
  gpt.final_norm.shift = assign(gpt.final_norm.shift, params['b'])

  # out head
  gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])


