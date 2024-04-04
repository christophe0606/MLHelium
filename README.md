# MLHelium



float32 and float16 Helium optimized kernels for ML inference and a few examples using those kernels.

If you need quantized kernels and/or use of an Arm NPU for Cortex-M, please use TensorFlow Lite for Micro.

This library is only focusing on float and small kernels with the goal of requiring the minimum of dependencies to run a model.

The model must be written in C (by calling the required kernels). No automatic conversion from TensorFlow or PyTorch is provided.

A script is provided to make it easier to export the weights from the Python and use them from the C.

