#pragma once

// model_i8 assumes DIM and HIDDEN DIM are mutliple of 4
#define DIM 288
#define HIDDEN_DIM 768
#define N_LAYERS 6
#define N_HEADS 6
#define N_KV_HEADS 6
#define VOCAB_SIZE 32000
#define MAX_SEQ_LEN 256
#define SHARED_WEIGHTS 1
#define KV_MUL  (N_HEADS / N_KV_HEADS) // integer multiplier of the kv sharing in multiquery
#define HEAD_SIZE  (DIM / N_HEADS)

// Used to for int8 quantized version
// model_i8 assumes DIM is a multiple of GROUP_SIZE
// and the matmul in model_i8 assumes GROUP_SIZE is 32
#define GROUP_SIZE 32
#define DIM_GS 9 // (DIM / GROUP_SIZE)

#define KV_DIM ((DIM * N_KV_HEADS) / N_HEADS)

// For accurate timing measurement, should be
// coherent with value used for MPS3
#define MPS3_FREQ 50000000UL