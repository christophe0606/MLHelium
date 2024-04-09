#ifndef ERROR_H_
#define ERROR_H_

enum {
  kNoError = 0,
  kErrorAllocRunState = -1,
  kNoMemoryBufferForCheckpoint = -2,
  kErrorBuildTokenizer = -3,
  kCannotEncodeNullText = -4,
  kEncodeError = -5,
  kBuildSamplerError = -6,
  kGenerateError = -7
} kError;

#endif