#pragma once

#define DIM 288
#define HIDDEN_DIM 768
#define N_LAYERS 6
#define N_HEADS 6
#define N_KV_HEADS 6
#define VOCAB_SIZE 32000
#define MAX_SEQ_LEN 256
#define SHARED_WEIGHTS 1

#define KV_DIM ((DIM * N_KV_HEADS) / N_HEADS)

