# simple

Demonstrate how to export a network from Python and read it from C.

The following command will create a `test.dat` containing a `f32` and `f16` network.

The tensors are aligned by default on `8` bytes boundary. It can be changed in the python function `serialize_tensors` that is having an argument `alignment`.

```shell
python create.py
```

Once the `test.dat` has been created, it can be loaded in the memory to be used by the C.

For small networks, the simplest is to use a C array. `test.dat` can be converted to a C array with:

```shell
python ../../scripts/convert_to_c.py -i test.dat network
```

This command will generate the 2 files `network.c` and `network.h`. By default, the network definition is using a macro `ALIGN_NETWORK` that will align the buffer on a `16` bytes boundary and ensuring that the tensor alignment inside the network buffer are respected.

This macro can be redefined.

`demo.cpp` is reading the two tensors and displaying their content.

`demo.cpp` is also showing how to copy a tensor to internal memory.

