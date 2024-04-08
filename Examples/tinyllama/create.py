import sys
import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer

sys.path.insert(0, '../../scripts')

from export import serialize_tensors,read_tensors

def load_checkpoint(checkpoint):

    # load the provided model checkpoint
    checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def append_f16(tensors,n):
    d = n.detach().cpu().view(-1).to(torch.float16).numpy()
    tensors.append(d)

def model_export(model, filepath):

    tensors = []

    
    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    

    print(f"""Network configuration to use in the C
dim = {p.dim}
hidden dim = {hidden_dim}
n_layers = {p.n_layers}
n_heads = {p.n_heads}
kv_heads = {n_kv_heads}
vocab_size = {p.vocab_size}
max_seq_len = {p.max_seq_len}
shared_classifier = {shared_classifier}
    """)

    # next write out the embedding weights
    append_f16(tensors,model.tok_embeddings.weight)

    # now all the layers
    # attention weights
    for layer in model.layers:
        append_f16(tensors,layer.attention_norm.weight)
    for layer in model.layers:
        append_f16(tensors, layer.attention.wq.weight)
    for layer in model.layers:
        append_f16(tensors, layer.attention.wk.weight)
    for layer in model.layers:
        append_f16(tensors, layer.attention.wv.weight)
    for layer in model.layers:
        append_f16(tensors, layer.attention.wo.weight)
    # ffn weights
    for layer in model.layers:
        append_f16(tensors, layer.ffn_norm.weight)
    for layer in model.layers:
        append_f16(tensors, layer.feed_forward.w1.weight)
    for layer in model.layers:
        append_f16(tensors, layer.feed_forward.w2.weight)
    for layer in model.layers:
        append_f16(tensors, layer.feed_forward.w3.weight)
    # final rmsnorm
    append_f16(tensors, model.norm.weight)
    # freqs_cis
    append_f16(tensors, model.freqs_cos[:p.max_seq_len])
    append_f16(tensors, model.freqs_sin[:p.max_seq_len])

    # final classifier weights
    if not shared_classifier:
        append_f16(tensors, model.output.weight)

    with open(filepath,"wb") as f:
        serialize_tensors(f,tensors)


model = load_checkpoint("../../Models/tinyllama/stories15M.pt")

model_export(model,"network.dat")