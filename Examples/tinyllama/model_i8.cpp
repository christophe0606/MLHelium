#if defined(WEIGHT_I8)
#include "model.h"

#include "dsp/matrix_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"
#include "dsp/complex_math_functions_f16.h"
#include "dsp/fast_math_functions_f16.h"

#include "dsp/matrix_functions.h"
#include "dsp/basic_math_functions.h"
#include "dsp/complex_math_functions.h"
#include "dsp/fast_math_functions.h"

#include "model_i8.h"

#include "common.h"
#include "memory.h"
#include "kernels.h"
#include "error.h"

#include <cstdio>

/*

Internal memory for some part of the transformer state

*/
#define ALIGNMENT_PAD (11*8)
//#define NB_INT_MEM (((DIM+N_HEADS * MAX_SEQ_LEN + VOCAB_SIZE + 4*DIM+2*HIDDEN_DIM + DIM + HIDDEN_DIM)*sizeof(FLOAT_TYPE)) + (DIM+HIDDEN_DIM))
#define NB_INT_MEM 16
static unsigned char* internal_mem[NB_INT_MEM+ALIGNMENT_PAD];
static memory_area_t internal ;

size_t get_internal_current_bytes()
{
    return internal.current_bytes;
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, FLOAT_TYPE* x, int n,int pos) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i+n*pos] * qx->s[i / GROUP_SIZE + DIM_GS * pos ];
    }
}

void quantize(QuantizedTensor *qx, FLOAT_TYPE* x, int n) {
    int num_groups = n / GROUP_SIZE;
    FLOAT_TYPE Q_MAX = (FLOAT_TYPE)127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        FLOAT_TYPE wmax = 0.0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            FLOAT_TYPE val = (FLOAT_TYPE)fabs(x[group * GROUP_SIZE + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        FLOAT_TYPE scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GROUP_SIZE; i++) {
            float16_t quant_value = x[group * GROUP_SIZE + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GROUP_SIZE + i] = quantized;
        }
    }
}

static int malloc_run_state(RunState* s) {

    s->token_embedding_table = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(DIM,FLOAT_TYPE);
    if (!s->token_embedding_table)
    {
        printf("Error mem alloc token_embedding_table\r\n");
    }

    s->x = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(DIM,FLOAT_TYPE);
    if (!s->x)
    {
        printf("Error mem alloc x\r\n");
    }
    s->xb = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(DIM,FLOAT_TYPE);
    if (!s->xb)
    {
        printf("Error mem alloc xb\r\n");
    }
    s->xb2 = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(DIM,FLOAT_TYPE);
    if (!s->xb2)
    {
        printf("Error mem alloc xb2\r\n");
    }

    
    s->hb = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(HIDDEN_DIM,FLOAT_TYPE);
    if (!s->hb)
    {
        printf("Error mem alloc hb\r\n");
    }
    s->hb2 = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(HIDDEN_DIM,FLOAT_TYPE);
    if (!s->hb2)
    {
        printf("Error mem alloc hb2\r\n");
    }

    s->xq.q = (int8_t*)MEM_INTERNAL_ALLOC(DIM,int8_t);
    if (!s->xq.q)
    {
        printf("Error mem alloc xq.q\r\n");
    }
    s->xq.s = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(DIM,FLOAT_TYPE);
    if (!s->xq.s)
    {
        printf("Error mem alloc xq.s\r\n");
    }

    s->hq.q = (int8_t*)MEM_INTERNAL_ALLOC(HIDDEN_DIM,int8_t);
    if (!s->hq.q)
    {
        printf("Error mem alloc hq.q\r\n");
    }
    s->hq.s = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(HIDDEN_DIM,FLOAT_TYPE);
    if (!s->hq.s)
    {
        printf("Error mem alloc hq.s\r\n");
    }

    s->q = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(DIM,FLOAT_TYPE);
    if (!s->q)
    {
        printf("Error mem alloc q\r\n");
    }

    // Caches are too big for internal memory so they are
    // allocated in the external heap
    s->key_cache = (FLOAT_TYPE*)ml_aligned_calloc(N_LAYERS * MAX_SEQ_LEN * KV_DIM, sizeof(FLOAT_TYPE));
    if (!s->key_cache)
    {
        printf("Error mem alloc key_cache\r\n");
    }
    s->value_cache = (FLOAT_TYPE*)ml_aligned_calloc(N_LAYERS * MAX_SEQ_LEN * KV_DIM, sizeof(FLOAT_TYPE));
    if (!s->value_cache)
    {
        printf("Error mem alloc value_cache\r\n");
    }

    s->att = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(N_HEADS * MAX_SEQ_LEN,FLOAT_TYPE);
    if (!s->att)
    {
        printf("Error mem alloc att\r\n");
    }

    s->logits = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(VOCAB_SIZE,FLOAT_TYPE);
    if (!s->logits)
    {
        printf("Error mem alloc logits\r\n");
    }

    s->cs_cache = (FLOAT_TYPE*)MEM_INTERNAL_ALLOC(DIM,FLOAT_TYPE);
    if (!s->cs_cache)
    {
        printf("Error mem alloc cs_cache\r\n");
    }

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits
     || !s->cs_cache
     || !s->xq.q || !s->xq.s 
     || !s->hq.q || !s->hq.s 
     || !s->token_embedding_table
     ) {
        return(kErrorAllocRunState);
    }
    return(kNoError);
}

static void free_run_state(RunState* s) {
    if (!s)
    {
        return;
    }
    // Using internal memory so no need to be released
    FREE_INTERNAL_ALLOC(s->token_embedding_table);
    FREE_INTERNAL_ALLOC(s->x);
    FREE_INTERNAL_ALLOC(s->xb);
    FREE_INTERNAL_ALLOC(s->xb2);
    FREE_INTERNAL_ALLOC(s->hb);
    FREE_INTERNAL_ALLOC(s->hb2);
    FREE_INTERNAL_ALLOC(s->q);
    FREE_INTERNAL_ALLOC(s->att);
    FREE_INTERNAL_ALLOC(s->logits);
    aligned_free(s->key_cache);
    aligned_free(s->value_cache);
}

static void memory_map_weights(TransformerWeights *w, const unsigned char* ptr) {
    // Get pointer to the weights inside the memory mapped network
    int ID = 0;
    w->q_tokens.q = get_i8_tensor(ptr,ID++);
    w->q_tokens.s = GET_FLOAT_TENSOR(ptr,ID++);
    
    for(int l=0;l < N_LAYERS; l++)
    {
       w->rms_att_weight[l] = GET_FLOAT_TENSOR(ptr,ID++);
    }

    for(int l=0;l < N_LAYERS; l++)
    {
       w->wq[l].q = get_i8_tensor(ptr,ID++);
       w->wq[l].s = GET_FLOAT_TENSOR(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wk[l].q = get_i8_tensor(ptr,ID++);
       w->wk[l].s = GET_FLOAT_TENSOR(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wv[l].q = get_i8_tensor(ptr,ID++);
       w->wv[l].s = GET_FLOAT_TENSOR(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wo[l].q = get_i8_tensor(ptr,ID++);
       w->wo[l].s = GET_FLOAT_TENSOR(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->rms_ffn_weight[l] = GET_FLOAT_TENSOR(ptr,ID++);
    }

    for(int l=0;l < N_LAYERS; l++)
    {
       w->w1[l].q = get_i8_tensor(ptr,ID++);
       w->w1[l].s = GET_FLOAT_TENSOR(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->w2[l].q = get_i8_tensor(ptr,ID++);
       w->w2[l].s = GET_FLOAT_TENSOR(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->w3[l].q = get_i8_tensor(ptr,ID++);
       w->w3[l].s = GET_FLOAT_TENSOR(ptr,ID++);
    }
    
    w->rms_final_weight = GET_FLOAT_TENSOR(ptr,ID++);

    w->freq_cos_sin = GET_FLOAT_TENSOR(ptr,ID++);

    if (SHARED_WEIGHTS)
    {
       w->wcls = w->q_tokens;
    }
    else 
    {
       w->wcls.q = get_i8_tensor(ptr,ID++);
       w->wcls.s = GET_FLOAT_TENSOR(ptr,ID++);
    }
}

static int read_checkpoint(const unsigned char* memory, TransformerWeights* weights) {
    if (!memory) { 
        return(kNoMemoryBufferForCheckpoint); 
    }
    

    memory_map_weights(weights, memory);
    return(kNoError);
}

int build_transformer(Transformer *t, const unsigned char* memory) {
    init_memory_area(&internal,(unsigned char*)internal_mem,NB_INT_MEM);

    // read in the Config and the Weights from the checkpoint
    int err = read_checkpoint(memory, &t->weights);
    if (err!=kNoError)
    {
        return(err);
    }
    // allocate the RunState buffers
    err = malloc_run_state(&t->state);
    return(err);
}

void free_transformer(Transformer* t) {
    // free the RunState buffers
    free_run_state(&t->state);
}



static void matmul(float16_t* xout, 
                   float16_t* x, 
                   float16_t* w, 
                   int cols, 
                   int rows) 
{
    arm_matrix_instance_f16 W;
    W.numRows = rows;
    W.numCols = cols;
    W.pData = w;


   
    arm_mat_vec_mult_f16(&W,x,xout);

}

static void matmul(float32_t* xout, 
                   float32_t* x, 
                   float32_t* w, 
                   int cols, 
                   int rows) 
{
    arm_matrix_instance_f32 W;
    W.numRows = rows;
    W.numCols = cols;
    W.pData = w;


   
    arm_mat_vec_mult_f32(&W,x,xout);

}


static void matmul(FLOAT_TYPE* xout, 
                   QuantizedTensor *x, 
                   QuantizedTensor *w, 
                   int n, 
                   int d) 
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    for (i = 0; i < d; i++) {

        FLOAT_TYPE val = (FLOAT_TYPE) 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GROUP_SIZE
        int j;
        for (j = 0; j <= n - GROUP_SIZE; j += GROUP_SIZE) {
            for (int k = 0; k < GROUP_SIZE; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((FLOAT_TYPE) ival) * w->s[(in + j) / GROUP_SIZE] * x->s[j / GROUP_SIZE];
            ival = 0;
        }

        xout[i] = val;
    }
}


FLOAT_TYPE* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    FLOAT_TYPE *x = s->x;
    int kv_mul = N_HEADS / N_KV_HEADS; // integer multiplier of the kv sharing in multiquery
    int head_size = DIM / N_HEADS;

    // copy the token embedding into x
    dequantize(&w->q_tokens, s->token_embedding_table, DIM,token);
    memcpy(x, s->token_embedding_table, DIM*sizeof(FLOAT_TYPE));

    // forward all the layers
    for(int l = 0; l < N_LAYERS; l++) {

        // attention rmsnorm
        RMS_NORM(s->xb, x, w->rms_att_weight[l], DIM);

        // key and value point to the kv cache
        int loff = l * MAX_SEQ_LEN * KV_DIM; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * KV_DIM;
        s->v = s->value_cache + loff + pos * KV_DIM;

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, DIM);
        matmul(s->q, &s->xq, &w->wq[l], DIM, DIM);
        matmul(s->k, &s->xq, &w->wk[l], DIM, KV_DIM);
        matmul(s->v, &s->xq, &w->wv[l], DIM, KV_DIM);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head

        for(int k=0;k<N_HEADS;k++)
        {
            memcpy(s->cs_cache+k*head_size,w->freq_cos_sin+head_size*pos,head_size*sizeof(float16_t));
        }
        CMPLX_MULT(s->cs_cache,s->q ,s->q,DIM>>1);
        CMPLX_MULT(s->cs_cache,s->k ,s->k,KV_DIM>>1);

        
        // multihead attention. iterate over all heads
        int h;

        for (h = 0; h < N_HEADS; h++) {
            // get the query vector for this head
            FLOAT_TYPE* q = s->q + h * head_size;
            // attention scores for this head
            FLOAT_TYPE* att = s->att + h * MAX_SEQ_LEN;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                FLOAT_TYPE* k = s->key_cache + loff + t * KV_DIM + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                FLOAT_TYPE score = (FLOAT_TYPE)0.0f;
                DOT_PROD(q,k,head_size,&score);

                score /= (FLOAT_COMPUTE)sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            SOFTMAX(att, pos + 1);

            // weighted sum of the values, store back into xb
            FLOAT_TYPE* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(FLOAT_TYPE));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                FLOAT_TYPE* v = s->value_cache + loff + t * KV_DIM + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                FLOAT_TYPE a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += (FLOAT_COMPUTE)a * (FLOAT_COMPUTE)v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, DIM);
        matmul(s->xb2, &s->xq, &w->wo[l], DIM, DIM);

        // residual connection back into x
        ADD(x,s->xb2,x,DIM);
        

        // ffn rmsnorm
        RMS_NORM(s->xb, x, w->rms_ffn_weight[l], DIM);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, DIM);
        matmul(s->hb, &s->xq, &w->w1[l], DIM, HIDDEN_DIM);
        matmul(s->hb2, &s->xq, &w->w3[l], DIM, HIDDEN_DIM);

        // SwiGLU non-linearity
        SWIGLU(s->hb, s->hb2,HIDDEN_DIM);

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, HIDDEN_DIM);
        matmul(s->xb, &s->hq, &w->w2[l], HIDDEN_DIM, DIM);

        // residual connection

        ADD(x,s->xb,x,DIM);
       
    }

    // final rmsnorm
    RMS_NORM(x, x, w->rms_final_weight, DIM);

    // classifier into logits
    quantize(&s->xq, x, DIM);
    matmul(s->logits, &s->xq, &w->wcls, DIM, VOCAB_SIZE);
    return (s->logits);
}

#endif