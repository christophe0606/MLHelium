# tinyllama

# How to install / build / run

It is an adaptation of the project `llama2.c` by Andrej Karpathy

https://github.com/karpathy/llama2.c/tree/master

You first must get the network checkpoint from Huggingface and the Tokenizer from the `llama2.c` project.

```shell
# Get the network
wget -O ../Models/tinyllama/stories15M.pt https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt?download=true 

# Get the tokenizer from https://github.com/karpathy/llama2.c
wget -O tok.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

The network must then be converted to f16 and serialized in a data format developed for those examples (simple and with buffer alignements):

```shell
python create.py
```

The file `net.bin` is generated. It can be loaded to memory.

This demo assumes the network is loaded at address `0x70000000` and the tokenizer at address `0x71D00000`.

When running on AVH, the files (`net.bin` and `tok.bin`) are read and written to those addresses in semi-hosting mode.

When running on MPS3, the files must be loaded during boot sequence.

For instance, the FPGA configuration file could be:

```
[IMAGES]
TOTALIMAGES: 3                     ;Number of Images (Max: 32)

IMAGE0ADDRESS: 0x01000000          ;Please select the required executable program
IMAGE0UPDATE: AUTO                 ;Image Update:NONE/AUTO/FORCE
IMAGE0FILE: \SOFTWARE\app.bin      ;Application

IMAGE1ADDRESS: 0x0C000000          ;Please select the required executable program
IMAGE1UPDATE: AUTO                 ;Image Update:NONE/AUTO/FORCE
IMAGE1FILE: \SOFTWARE\net_int8.bin ;Network with int8 weight and float scalings

IMAGE2ADDRESS: 0x0DD00000          ;Please select the required executable program
IMAGE2UPDATE: AUTO                 ;Image Update:NONE/AUTO/FORCE
IMAGE2FILE: \SOFTWARE\tok.bin      ;Tokenizer



```

Building is relying on CMSIS build tools.

# Changes

The file `model.py` was changed to interleave `cos` and `sin` samples so that Helium complex number instructions can be used for the RoPE.

The `demo.cpp` is an adaptation from the original `run.cpp` of the `llama2.c` project but lots of things were changed:

- Network and tokenizer are memory mapped
- Use the file format developed for those examples
- Tensor and working memory is aligned
- Some working buffers are placed into internal memory
- chat mode removed
- CMSIS-DSP f16 mode is used
- The start seed is constant

There may be some bugs that has been introduced by those changes.

It is an example and not something to use in a product.

## Rebuilding

To rebuild with CMSIS build tools, you must change the variables in the csolution file.

The `DSPPP` variable must be path pointing to the CMSIS-DSP C++ template header files (they are not part of the CMSIS-DSP pack yet so must be got from the github repository of CMSIS-DSP)
