#pragma once

#include <cstdlib>
#include <cstdint>

#define SAFE_FREE(x)\
  if ((x))          \
  {                 \
     free((x));     \
  }

extern void * ml_calloc( size_t elementCount, size_t elementSize );
extern void * ml_malloc( size_t bytes );
extern void * ml_aligned_calloc( size_t elementCount, size_t elementSize );
extern void * ml_aligned_malloc( size_t bytes );
extern uint32_t get_mem_usage();