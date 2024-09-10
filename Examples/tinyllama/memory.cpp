#include "memory.h"
#include "common.h"

static uint32_t mem_usage = 0;

uint32_t get_mem_usage()
{
   return mem_usage;
}

void * ml_calloc( size_t elementCount, size_t elementSize )
{
    mem_usage += elementCount * elementSize;
    return(calloc(elementCount,elementSize));
}

void * ml_malloc( size_t bytes )
{
    mem_usage += bytes;
    return(malloc(bytes));
}

void * ml_aligned_calloc( size_t elementCount, size_t elementSize )
{
    size_t allocated_memory;
    void* res = aligned_malloc(8, elementCount*elementSize,&allocated_memory);

    mem_usage += allocated_memory;
    return(res);
}

void * ml_aligned_malloc( size_t bytes )
{
    size_t allocated_memory;
    void* res = aligned_malloc(8, bytes,&allocated_memory);

    mem_usage += allocated_memory;
    return(res);
}