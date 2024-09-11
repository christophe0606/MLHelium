#if defined(WEIGHT_F8)
#include "model.h"

#include "dsp/matrix_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"
#include "dsp/complex_math_functions_f16.h"
#include "dsp/fast_math_functions_f16.h"

#include "model_f8.h"

#include "common.h"
#include "memory.h"
#include "kernels.h"
#include "error.h"

/*

Internal memory for some part of the transformer state

*/
#define NB_INT_MEM ((DIM+N_HEADS * MAX_SEQ_LEN + VOCAB_SIZE + 4*DIM+2*HIDDEN_DIM)*sizeof(float16_t))
static unsigned char* internal_mem[NB_INT_MEM];
static memory_area_t internal ;

size_t get_internal_current_bytes()
{
    return internal.current_bytes;
}

static int malloc_run_state(RunState* s) {

    
    s->x = (float16_t*)add_aligned_buffer_to_area(&internal,DIM*sizeof(float16_t),8);
    s->xb = (float16_t*)add_aligned_buffer_to_area(&internal,DIM*sizeof(float16_t),8);
    s->xb2 = (float16_t*)add_aligned_buffer_to_area(&internal,DIM*sizeof(float16_t),8);

    
    s->hb = (float16_t*)add_aligned_buffer_to_area(&internal,HIDDEN_DIM*sizeof(float16_t),8);
    s->hb2 = (float16_t*)add_aligned_buffer_to_area(&internal,HIDDEN_DIM*sizeof(float16_t),8);

    s->q = (float16_t*)add_aligned_buffer_to_area(&internal,DIM*sizeof(float16_t),8);

    // Caches are too big for internal memory so they are
    // allocated in the external heap
    s->key_cache = (float16_t*)ml_aligned_calloc(N_LAYERS * MAX_SEQ_LEN * KV_DIM, sizeof(float16_t));
    s->value_cache = (float16_t*)ml_aligned_calloc(N_LAYERS * MAX_SEQ_LEN * KV_DIM, sizeof(float16_t));
    
    s->att = (float16_t*)add_aligned_buffer_to_area(&internal,N_HEADS * MAX_SEQ_LEN*sizeof(float16_t),8);

    s->logits = (float16_t*)add_aligned_buffer_to_area(&internal,VOCAB_SIZE*sizeof(float16_t),8);

    s->cs_cache = (float16_t*)add_aligned_buffer_to_area(&internal,DIM*sizeof(float16_t),8);

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits
     || !s->cs_cache) {
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
    //aligned_free(s->x);
    //aligned_free(s->xb);
    //aligned_free(s->xb2);
    //aligned_free(s->hb);
    //aligned_free(s->hb2);
    //aligned_free(s->q);
    //aligned_free(s->att);
    //aligned_free(s->logits);
    aligned_free(s->key_cache);
    aligned_free(s->value_cache);
}

static void memory_map_weights(TransformerWeights *w, const unsigned char* ptr) {
    // Get pointer to the weights inside the memory mapped network
    int ID = 0;
    w->token_embedding_table = get_f8_tensor(ptr,ID++);
    for(int l=0;l < N_LAYERS; l++)
    {
       w->rms_att_weight[l] = get_f8_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wq[l] = get_f8_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wk[l] = get_f8_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wv[l] = get_f8_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wo[l] = get_f8_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->rms_ffn_weight[l] = get_f8_tensor(ptr,ID++);
    }

    for(int l=0;l < N_LAYERS; l++)
    {
       w->w1[l] = get_f8_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->w2[l] = get_f8_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->w3[l] = get_f8_tensor(ptr,ID++);
    }
    
    w->rms_final_weight = get_f8_tensor(ptr,ID++);

    w->freq_cos_sin = get_f8_tensor(ptr,ID++);

    w->wcls = SHARED_WEIGHTS ? w->token_embedding_table : get_f8_tensor(ptr,ID++);
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


static void matmul(float16_t* xout, float16_t* x, float8_t* w, int cols, int rows) {
    arm_matrix_instance_f8 W;
    W.numRows = rows;
    W.numCols = cols;
    W.pData = w;


   
    arm_mat_vec_mult_f8_f16(&W,x,xout);

}


float16_t* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float16_t *x = s->x;
    int kv_mul = N_HEADS / N_KV_HEADS; // integer multiplier of the kv sharing in multiquery
    int head_size = DIM / N_HEADS;

    // copy the token embedding into x
    float8_t* content_row = w->token_embedding_table + token * DIM;
    arm_copy_f8_to_f16(x, content_row, DIM);

    // forward all the layers
    for(int l = 0; l < N_LAYERS; l++) {

        // attention rmsnorm
        arm_rms_norm_f8_f16(s->xb, x, w->rms_att_weight[l], DIM);

        // key and value point to the kv cache
        int loff = l * MAX_SEQ_LEN * KV_DIM; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * KV_DIM;
        s->v = s->value_cache + loff + pos * KV_DIM;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq[l], DIM, DIM);
        matmul(s->k, s->xb, w->wk[l], DIM, KV_DIM);
        matmul(s->v, s->xb, w->wv[l], DIM, KV_DIM);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head

        for(int k=0;k<N_HEADS;k++)
        {
            memcpy(s->cs_cache+k*head_size,w->freq_cos_sin+head_size*pos,head_size*sizeof(float16_t));
        }
        arm_cmplx_mult_cmplx_f16(s->cs_cache,s->q ,s->q,DIM>>1);
        arm_cmplx_mult_cmplx_f16(s->cs_cache,s->k ,s->k,KV_DIM>>1);

        
        // multihead attention. iterate over all heads
        int h;

        for (h = 0; h < N_HEADS; h++) {
            // get the query vector for this head
            float16_t* q = s->q + h * head_size;
            // attention scores for this head
            float16_t* att = s->att + h * MAX_SEQ_LEN;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float16_t* k = s->key_cache + loff + t * KV_DIM + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float16_t score = 0.0f16;
                arm_dot_prod_f16(q,k,head_size,&score);

                score /= (_Float16)sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            arm_softmax_f16(att, pos + 1);

            // weighted sum of the values, store back into xb
            float16_t* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float16_t));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float16_t* v = s->value_cache + loff + t * KV_DIM + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float16_t a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += (_Float16)a * (_Float16)v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo[l], DIM, DIM);

        // residual connection back into x
        arm_add_f16(x,s->xb2,x,DIM);
        

        // ffn rmsnorm
        arm_rms_norm_f8_f16(s->xb, x, w->rms_ffn_weight[l], DIM);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1[l], DIM, HIDDEN_DIM);
        matmul(s->hb2, s->xb, w->w3[l], DIM, HIDDEN_DIM);

        // SwiGLU non-linearity
        arm_swiglu_f16(s->hb, s->hb2,HIDDEN_DIM);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2[l], HIDDEN_DIM, DIM);

        // residual connection

        arm_add_f16(x,s->xb,x,DIM);
       
    }

    // final rmsnorm
    arm_rms_norm_f8_f16(x, x, w->rms_final_weight, DIM);

    // classifier into logits
    matmul(s->logits, x, w->wcls, DIM, VOCAB_SIZE);
    return (s->logits);
}

#endif