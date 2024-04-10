# tinyllama

```shell
# Get the network
wget -O ../Models/tinyllama/stories15M.pt https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt?download=true 

# Get the tokenizer from https://github.com/karpathy/llama2.c
wget -O tok.bin https://github.com/4bbec843-d0db-4c21-a977-096c983faab7
```

Converting the network to f16:

```shell
python create.py
```

The file `net.bin` is generated. It can be loaded to memory.

This demo assumes the network is loaded at address `0x70000000` and the tokenizer at address `0x71D00000`.

When running on AVH, the files (`net.bin` and `tok.bin`) are read and written to those addresses.

When running on MPS3, the files must be loaded during boot sequence.

