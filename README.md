# MLHelium

Experiment with ML using Helium f16 instructions. It relies on CMSIS-DSP but a few additional kernels may be provided by this library when not available in CMSIS-DSP.

It is **NOT** a new ML framework. It is just some examples to demonstrate that for small simple networks, CMSIS-DSP may be enough (but requires more work).

This library is only focusing on float and small kernels with the goal of requiring the minimum of dependencies to run a model.

If you need quantized kernels and/or use of an Arm NPU for Cortex-M, please use TensorFlow Lite for Micro and CMSIS-NN.

The model must be written in C (by calling the required kernels). No automatic conversion from TensorFlow or PyTorch is provided.

A script is provided to make it easier to export the weights from the Python and use them from the C.

LICENSE is Apache except for some examples that have been adapted from github repository and may use other licences (like MIT).

