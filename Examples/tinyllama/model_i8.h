#pragma once

#include "arm_math_types_f16.h"

/*
#define FLOAT_TYPE float16_t
#define FLOAT_COMPUTE _Float16
#define RMS_NORM arm_rms_norm_f16
#define CMPLX_MULT arm_cmplx_mult_cmplx_f16
#define ADD arm_add_f16
#define DOT_PROD arm_dot_prod_f16
#define GET_FLOAT_TENSOR get_f16_tensor
#define SOFTMAX arm_softmax_f16
#define SWIGLU arm_swiglu_f16
#define MAX_VEC arm_max_f16
*/

#define FLOAT_TYPE float32_t
#define FLOAT_COMPUTE float32_t
#define RMS_NORM arm_rms_norm_f32
#define CMPLX_MULT arm_cmplx_mult_cmplx_f32
#define ADD arm_add_f32
#define DOT_PROD arm_dot_prod_f32
#define GET_FLOAT_TENSOR get_f32_tensor
#define SOFTMAX_MIXED(A,B) arm_softmax_f32_f16(A,B,s->tmp)
#define SOFTMAX arm_softmax_f32
#define SWIGLU arm_swiglu_f32
#define MAX_VEC arm_max_f32


#define MEM_ALLOC(R,NB,DT,ISINT)\
if (ISINT)                                                    \
{                                                           \
    R=(DT*)add_aligned_buffer_to_area(&internal,(NB)*sizeof(DT),8);\
}                                                           \
else                                                        \
{                                                           \
   R=(DT*)ml_aligned_calloc((NB),sizeof(DT));                      \
}

#define MEM_FREE(P,ISINT)\
if (!ISINT)              \
{                      \
    aligned_free((P)); \
}

typedef struct {
    int8_t* q;    // quantized values
    FLOAT_TYPE* s; // scaling factors
} QuantizedTensor;


// ----------------------------------------------------------------------------
// Transformer model

struct TransformerWeights {
    // token embedding table
    QuantizedTensor q_tokens;    // (vocab_size, dim)
    // weights for rmsnorms
    float32_t* rms_att_weight[N_LAYERS]; // (layer, dim) rmsnorm weights
    float32_t* rms_ffn_weight[N_LAYERS]; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor wq[N_LAYERS]; // (layer, dim, n_heads * head_size)
    QuantizedTensor wk[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor wv[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor wo[N_LAYERS]; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor w1[N_LAYERS]; // (layer, hidden_dim, dim)
    QuantizedTensor w2[N_LAYERS]; // (layer, dim, hidden_dim)
    QuantizedTensor w3[N_LAYERS]; // (layer, hidden_dim, dim)
    // final rmsnorm
    float32_t* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor wcls;

    // Precomputed sin and cos frequencies
    // cos sin interleaved
    FLOAT_TYPE *freq_cos_sin;

} ;

struct RunState {
    // current wave of activations
    FLOAT_TYPE* token_embedding_table;    // dim
    FLOAT_TYPE* x; // activation at current time stamp (dim,)
    FLOAT_TYPE* xb; // same, but inside a residual branch (dim,)
    FLOAT_TYPE* xb2; // an additional buffer just for convenience (dim,)
    FLOAT_TYPE* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    FLOAT_TYPE* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    FLOAT_TYPE* q; // query (dim,)
    FLOAT_TYPE* k; // key (dim,)
    FLOAT_TYPE* v; // value (dim,)
    FLOAT_TYPE* att; // buffer for scores/attention values (n_heads, seq_len)
    FLOAT_TYPE* logits; // output logits
    // kv cache
    FLOAT_TYPE* key_cache;   // (layer, seq_len, dim)
    FLOAT_TYPE* value_cache; // (layer, seq_len, dim)

    // cos_sin cache 
    FLOAT_TYPE *cs_cache;

    float16_t* tmp; // buffer for scores/attention values (n_heads, seq_len)


} ;

struct Transformer {
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
} ;

extern size_t get_internal_current_bytes();
extern void free_transformer(Transformer* t);
extern int build_transformer(Transformer *t, const unsigned char* memory);
extern FLOAT_TYPE* forward(Transformer* transformer, int token, int pos);
