#pragma once

#include "arm_math_types_f16.h"
// Define float8 datatype
#include "common.h"

#define FLOAT_TYPE float16_t
#define MAX_VEC arm_max_f16
#define SOFTMAX arm_softmax_f16

// ----------------------------------------------------------------------------
// Transformer model with float8 heights

struct TransformerWeights {
    // token embedding table
    float8_t* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float8_t* rms_att_weight[N_LAYERS]; // (layer, dim) rmsnorm weights
    float8_t* rms_ffn_weight[N_LAYERS]; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float8_t* wq[N_LAYERS]; // (layer, dim, n_heads * head_size)
    float8_t* wk[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    float8_t* wv[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    float8_t* wo[N_LAYERS]; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float8_t* w1[N_LAYERS]; // (layer, hidden_dim, dim)
    float8_t* w2[N_LAYERS]; // (layer, dim, hidden_dim)
    float8_t* w3[N_LAYERS]; // (layer, hidden_dim, dim)
    // final rmsnorm
    float8_t* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float8_t* wcls;

    // Precomputed sin and cos frequencies
    // cos sin interleaved
    float8_t *freq_cos_sin;

} ;

struct RunState {
    // current wave of activations
    float16_t* x; // activation at current time stamp (dim,)
    float16_t* xb; // same, but inside a residual branch (dim,)
    float16_t* xb2; // an additional buffer just for convenience (dim,)
    float16_t* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float16_t* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float16_t* q; // query (dim,)
    float16_t* k; // key (dim,)
    float16_t* v; // value (dim,)
    float16_t* att; // buffer for scores/attention values (n_heads, seq_len)
    float16_t* logits; // output logits
    // kv cache
    float16_t* key_cache;   // (layer, seq_len, dim)
    float16_t* value_cache; // (layer, seq_len, dim)

    // cos_sin cache 
    float16_t *cs_cache;
} ;

struct Transformer {
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} ;

extern size_t get_internal_current_bytes();
extern void free_transformer(Transformer* t);
extern int build_transformer(Transformer *t, const unsigned char* memory);
extern float16_t* forward(Transformer* transformer, int token, int pos);

