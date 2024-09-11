import sys
import numpy as np
import torch
import argparse

from torch import nn

from model import ModelArgs, Transformer

sys.path.insert(0, '../../scripts')

from export import serialize_tensors,read_tensors


parser = argparse.ArgumentParser(description='Convert network')
parser.add_argument('-f8', action='store_true', help="f8 weights")
parser.add_argument('-i8', action='store_true', help="int8 weights")
args = parser.parse_args()

if args.i8:
    FLOAT16_TYPE = False
else:
    FLOAT16_TYPE = True

if args.f8:
    print("f8 weights (2 bits of mantissa)")

if args.i8:
    print("int8 weights")

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

def append_f32(tensors,n):
    d = n.detach().cpu().view(-1).to(torch.float32).numpy()
    tensors.append(d)

def append_int8(tensors,n):
    d = n.detach().cpu().view(-1).to(torch.int8).numpy()
    tensors.append(d)

def array_f16_to_f8(x):
    r = x.tobytes()[1::2]
    r = np.frombuffer(r,dtype='int8')
    return(r)

# Shift left with widening on Helium
# is reading vector in an interleaved way
# Bottom with get the even lanes and Top will get the
# # odd lanes.
# So we pre-interleave the data so that it is read in right
# order on Helium without consuming too many cycles to do
# the reordering
def interleave(a):
   if len(a)<16:
       r = 16-len(a) 
       a = np.pad(a, (0,r), 'constant', constant_values=0)
   nb = len(a)
   b = np.zeros(len(a),dtype=np.int8)
   b[0::2] = a[:len(b[0::2])]
   b[1::2] = a[8:len(b[1::2])+8]
   return(b)

def append_f8(tensors,n):
    d = n.detach().cpu().view(-1).to(torch.float16).numpy()
    r = array_f16_to_f8(d)
    r = interleave(r)
    tensors.append(r)

def append_float_tensor(tensors,n):
    if args.f8:
        append_f8(tensors,n)
    else:
        if FLOAT16_TYPE:
           append_f16(tensors,n)
        else:
           append_f32(tensors,n)

def model_float_export(model, filepath):

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
    append_float_tensor(tensors,model.tok_embeddings.weight)

    # now all the layers
    # attention weights
    for layer in model.layers:
        append_float_tensor(tensors,layer.attention_norm.weight)
    for layer in model.layers:
        append_float_tensor(tensors, layer.attention.wq.weight)
    for layer in model.layers:
        append_float_tensor(tensors, layer.attention.wk.weight)
    for layer in model.layers:
        append_float_tensor(tensors, layer.attention.wv.weight)
    for layer in model.layers:
        append_float_tensor(tensors, layer.attention.wo.weight)
    # ffn weights
    for layer in model.layers:
        append_float_tensor(tensors, layer.ffn_norm.weight)
    for layer in model.layers:
        append_float_tensor(tensors, layer.feed_forward.w1.weight)
    for layer in model.layers:
        append_float_tensor(tensors, layer.feed_forward.w2.weight)
    for layer in model.layers:
        append_float_tensor(tensors, layer.feed_forward.w3.weight)
    # final rmsnorm
    append_float_tensor(tensors, model.norm.weight)
    # freqs_cis
    # Interleave cos/sin to be able to use Helium complex instructions
    s = model.freqs_cos[:p.max_seq_len].shape
    cs = torch.zeros((s[0],2*s[1]))
    cs[:,0::2] = model.freqs_cos[:p.max_seq_len]
    cs[:,1::2] = model.freqs_sin[:p.max_seq_len]
    append_float_tensor(tensors, cs)

    # final classifier weights
    if not shared_classifier:
        append_float_tensor(tensors, model.output.weight)

    with open(filepath,"wb") as f:
        serialize_tensors(f,tensors)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

def check_weight(w,i,group_size):
    assert w.numel() % group_size == 0, f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

def quantize_export(tensors,w,i,group_size):
    check_weight(w,i,group_size)
    q, s, err = quantize_q80(w, group_size)
    append_int8(tensors, q) # save the tensor in int8
    append_float_tensor(tensors, s) # save scale factors

def model_int8_export(model, filepath):

    tensors = []

    group_size = 64

    
    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    if not shared_classifier:
        p.vocab_size = -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads

    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    
    

    print(f"""Network configuration to use in the C
dim = {p.dim}
hidden dim = {hidden_dim}
n_layers = {p.n_layers}
n_heads = {p.n_heads}
kv_heads = {n_kv_heads}
vocab_size = {p.vocab_size}
max_seq_len = {p.max_seq_len}
shared_classifier = {shared_classifier}
group_size = {group_size}
    """)

    # next write out the embedding weights
    quantize_export(tensors,model.tok_embeddings.weight,0,group_size)

    # now all the layers
    # attention weights
    for layer in model.layers:
        append_float_tensor(tensors,layer.attention_norm.weight)

    for layer in model.layers:
        quantize_export(tensors, layer.attention.wq.weight,1,group_size)

    for layer in model.layers:
        quantize_export(tensors, layer.attention.wk.weight,2,group_size)

    for layer in model.layers:
        quantize_export(tensors, layer.attention.wv.weight,3,group_size)

    for layer in model.layers:
        quantize_export(tensors, layer.attention.wo.weight,4,group_size)

    # ffn weights
    for layer in model.layers:
        append_float_tensor(tensors, layer.ffn_norm.weight)

    for layer in model.layers:
        quantize_export(tensors, layer.feed_forward.w1.weight,5,group_size)

    for layer in model.layers:
        quantize_export(tensors, layer.feed_forward.w2.weight,6,group_size)

    for layer in model.layers:
        quantize_export(tensors, layer.feed_forward.w3.weight,7,group_size)

    # final rmsnorm
    append_float_tensor(tensors, model.norm.weight)
    # freqs_cis
    # Interleave cos/sin to be able to use Helium complex instructions
    s = model.freqs_cos[:p.max_seq_len].shape
    cs = torch.zeros((s[0],2*s[1]))
    cs[:,0::2] = model.freqs_cos[:p.max_seq_len]
    cs[:,1::2] = model.freqs_sin[:p.max_seq_len]
    append_float_tensor(tensors, cs)

    # final classifier weights
    if not shared_classifier:
        quantize_export(tensors, model.output.weight,8,group_size)

    with open(filepath,"wb") as f:
        serialize_tensors(f,tensors)

model = load_checkpoint("../../Models/tinyllama/stories15M.pt")

if args.f8:
    name = "net_f8.bin"
elif args.i8:
    name = "net_int8.bin"
else:
    name = "net_f16.bin"

if args.i8:
   model_int8_export(model,name)
else:
   model_float_export(model,name)