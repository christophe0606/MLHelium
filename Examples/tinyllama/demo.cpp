/* Inference for Llama-2 Transformer model in pure C 
 * 
 * Modified from https://github.com/karpathy/llama2.c to:
 * - Use f16
 * - Use memory mapped files (and the file format used in this project)
 * - Use CMSIS-DSP
 * - Use Helium
 * - Chat removed
 * - Use CMSIS build tools
 * 
 * Original Copyright (c) 2023 Andrej 
 * 
 */

#include "RTE_Components.h"
#include  CMSIS_device_header

#if defined(MPS3)
#include "stdout_USART.h"
#endif

#include <iostream>
#include "common.h"
#include "error.h"
#include <cstdio>

#include <cstdio>
#include <cstdlib>
//#include <ctype>
#include <time.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>

#include "dsp/matrix_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"
#include "dsp/complex_math_functions_f16.h"
#include "dsp/fast_math_functions_f16.h"


#include "kernels.h"

#define DIM 288
#define HIDDEN_DIM 768
#define N_LAYERS 6
#define N_HEADS 6
#define N_KV_HEADS 6
#define VOCAB_SIZE 32000
#define MAX_SEQ_LEN 256
#define SHARED_WEIGHTS 1

#define KV_DIM ((DIM * N_KV_HEADS) / N_HEADS)

/*

Internal memory for some part of the transformer state

*/
#define NB_INT_MEM ((DIM+N_HEADS * MAX_SEQ_LEN + VOCAB_SIZE + 4*DIM+2*HIDDEN_DIM)*sizeof(float16_t))
static unsigned char* internal_mem[NB_INT_MEM];
static memory_area_t internal ;

extern "C" {
#if defined(MPS3)
    #include "disp_token.h"
    #include "Board_GLCD.h"
#endif
    extern void demo();
}

#define SAFE_FREE(x)\
  if ((x))          \
  {                 \
     free((x));     \
  }

static uint32_t mem_usage = 0;

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

// ----------------------------------------------------------------------------
// Transformer model

struct TransformerWeights {
    // token embedding table
    float16_t* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float16_t* rms_att_weight[N_LAYERS]; // (layer, dim) rmsnorm weights
    float16_t* rms_ffn_weight[N_LAYERS]; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float16_t* wq[N_LAYERS]; // (layer, dim, n_heads * head_size)
    float16_t* wk[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    float16_t* wv[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    float16_t* wo[N_LAYERS]; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float16_t* w1[N_LAYERS]; // (layer, hidden_dim, dim)
    float16_t* w2[N_LAYERS]; // (layer, dim, hidden_dim)
    float16_t* w3[N_LAYERS]; // (layer, hidden_dim, dim)
    // final rmsnorm
    float16_t* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float16_t* wcls;

    // Precomputed sin and cos frequencies
    // cos sin interleaved
    float16_t *freq_cos_sin;

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

int malloc_run_state(RunState* s) {

    
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

void free_run_state(RunState* s) {
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

void memory_map_weights(TransformerWeights *w, const unsigned char* ptr) {
    // Get pointer to the weights inside the memory mapped network
    int ID = 0;
    w->token_embedding_table = get_f16_tensor(ptr,ID++);
    for(int l=0;l < N_LAYERS; l++)
    {
       w->rms_att_weight[l] = get_f16_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wq[l] = get_f16_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wk[l] = get_f16_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wv[l] = get_f16_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->wo[l] = get_f16_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->rms_ffn_weight[l] = get_f16_tensor(ptr,ID++);
    }

    for(int l=0;l < N_LAYERS; l++)
    {
       w->w1[l] = get_f16_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->w2[l] = get_f16_tensor(ptr,ID++);
    }
    for(int l=0;l < N_LAYERS; l++)
    {
       w->w3[l] = get_f16_tensor(ptr,ID++);
    }
    
    w->rms_final_weight = get_f16_tensor(ptr,ID++);

    w->freq_cos_sin = get_f16_tensor(ptr,ID++);

    w->wcls = SHARED_WEIGHTS ? w->token_embedding_table : get_f16_tensor(ptr,ID++);
}

int read_checkpoint(const unsigned char* memory, TransformerWeights* weights) {
    if (!memory) { 
        return(kNoMemoryBufferForCheckpoint); 
    }
    

    memory_map_weights(weights, memory);
    return(kNoError);
}

int build_transformer(Transformer *t, const unsigned char* memory) {
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

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer




void matmul(float16_t* xout, float16_t* x, float16_t* w, int cols, int rows) {
    arm_matrix_instance_f16 W;
    W.numRows = rows;
    W.numCols = cols;
    W.pData = w;


   
    arm_mat_vec_mult_f16(&W,x,xout);

}


float16_t* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float16_t *x = s->x;
    int kv_mul = N_HEADS / N_KV_HEADS; // integer multiplier of the kv sharing in multiquery
    int head_size = DIM / N_HEADS;

    // copy the token embedding into x
    float16_t* content_row = w->token_embedding_table + token * DIM;
    memcpy(x, content_row, DIM*sizeof(float16_t));

    // forward all the layers
    for(int l = 0; l < N_LAYERS; l++) {

        // attention rmsnorm
        arm_rms_norm_f16(s->xb, x, w->rms_att_weight[l], DIM);

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
        arm_rms_norm_f16(s->xb, x, w->rms_ffn_weight[l], DIM);

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
    arm_rms_norm_f16(x, x, w->rms_final_weight, DIM);

    // classifier into logits
    matmul(s->logits, x, w->wcls, DIM, VOCAB_SIZE);
    return (s->logits);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    const char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

template<typename T>
T interpret(const unsigned char* p)
{
    uint32_t v;
    /* Data may be unaligned in the tokenizer buffer */
    memcpy(&v,p,4);

    return(*reinterpret_cast<const T*>(&v));
}

int build_tokenizer(Tokenizer* t, const unsigned char* memory, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)ml_malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)ml_malloc(vocab_size * sizeof(float));
    if (!t->vocab || !t->vocab_scores)
    {
        return(kErrorBuildTokenizer);
    }
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    if (!memory) { 
        return(kNoMemoryBufferForCheckpoint); 
    }
    t->max_token_length = interpret<int>(memory);
    memory += 4;

    int len;
    for (int i = 0; i < vocab_size; i++) {
        t->vocab_scores[i] = interpret<float>(memory);
        memory += 4;

        len = interpret<int>(memory);
        // len may not be a multiple of 4 bytes
        // so other values are not aligned when read in the memory
        // buffer
        memory += 4;

        t->vocab[i] = (char *)ml_malloc(len + 1);
        if (!t->vocab[i])
        {
            return(kErrorBuildTokenizer);
        }
        memcpy(t->vocab[i],memory,len);
        memory += len;

        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    return(kNoError);
}

void free_tokenizer(Tokenizer* t) {
    if (!t)
    {
        return;
    }
    for (int i = 0; i < t->vocab_size; i++) { SAFE_FREE(t->vocab[i]); }
    SAFE_FREE(t->vocab);
    SAFE_FREE(t->vocab_scores);
    SAFE_FREE(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

#if defined(MPS3)
static char msg[256];
#endif

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    if ((piece[0]=='\n') && (piece[1]=='\0'))
    {
#if !defined(MPS3)
       printf("\r\n");
#else
       disp_token("\r\n");
#endif
    }
    else 
    {
#if !defined(MPS3)
       printf("%s", piece);
#else
       disp_token(piece);
#endif
    }
}

int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str, .id = 0 }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

int encode(Tokenizer* t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { 
        return(kCannotEncodeNullText); 
    }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*)ml_malloc(t->vocab_size * sizeof(TokenIndex));
        if (!t->sorted_vocab) 
        { 
            return(kEncodeError); 
        }
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*)ml_malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    if (!str_buffer) 
    { 
            return(kEncodeError); 
    }

    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point Last code point Byte 1  Byte 2  Byte 3  Byte 4
    // U+0000 U+007F      0xxxxxxx
    // U+0080 U+07FF      110xxxxx  10xxxxxx
    // U+0800 U+FFFF      1110xxxx  10xxxxxx  10xxxxxx
    // U+10000  U+10FFFF    11110xxx  10xxxxxx  10xxxxxx  10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (unsigned int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    SAFE_FREE(str_buffer);

    return(kNoError);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

struct Sampler {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} ;

int sample_argmax(float16_t* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float16_t* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float16_t* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

int build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex*)ml_malloc(sampler->vocab_size * sizeof(ProbIndex));
    if (!sampler->probindex) 
    { 
        return(kBuildSamplerError); 
    }
    return(kNoError);
}

void free_sampler(Sampler* sampler) {
    if (!sampler)
    {
        return;
    }
    SAFE_FREE(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float16_t* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        arm_softmax_f16(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

volatile uint32_t nb_sys_tick_round=0;
#define MAX_SYSTICK 0xFFFFFFUL

extern "C" {
    void SysTick_Handler(void);
}

void SysTick_Handler(void)  {                               /* SysTick interrupt Handler. */
  nb_sys_tick_round++;                                                /* See startup file startup_LPC17xx.s for SysTick vector */ 
}
   

extern "C" uint32_t SystemCoreClock;


float time_in_ms() {
    return(1000.0*(MAX_SYSTICK-SysTick->VAL + MAX_SYSTICK * nb_sys_tick_round)/SystemCoreClock);
}

long time_in_cycles() {
    return(MAX_SYSTICK - SysTick->VAL + MAX_SYSTICK * nb_sys_tick_round);
}


// ----------------------------------------------------------------------------
// generation loop

int generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)ml_malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    if (!prompt_tokens) 
    { 
        return(kGenerateError); 
    }

    int err = encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (err != kNoError)
    {
        return(err);
    }
    if (num_prompt_tokens < 1) {
        return(kGenerateError);
    }

    // start the main loop
    float start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float16_t* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\r\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        float end = time_in_ms();
        printf("\r\nduration: %f ms\n", (double)(end-start));
        printf("\r\nachieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
#if defined(MPS3)
        sprintf(msg,"\r\n\nduration: %f ms\r\n", (double)(end-start));
        disp_token(msg);
        sprintf(msg,"achieved tok/s: %f\r\n", (pos-1) / (double)(end-start)*1000);
        disp_token(msg);
#endif

    }

    SAFE_FREE(prompt_tokens);
    return(kNoError);
}



const unsigned char* load_mem(const char* checkpoint,const unsigned char* refbuf)
{
    std::ifstream fin(checkpoint, std::ios::binary);
    if (!fin) 
    {
        return(nullptr);
    }

    std::streampos begin,end;
    begin = fin.tellg();
    fin.seekg (0, std::ios::end);
    end = fin.tellg();
    fin.seekg (0, std::ios::beg);

    printf("%s : %lld bytes (%llX)\r\n",checkpoint,end-begin,end-begin);
    const unsigned char *buf;
    if (refbuf!=nullptr)
    {
       buf = refbuf;
    }
    else
    {
       buf =  (const unsigned char*)ml_malloc(end-begin);
    }
    if (buf == nullptr)
    {
        printf("Not enough memory\r\n");
        return(nullptr);
    }

    fin.read(const_cast<char*>(reinterpret_cast<const char*>(buf)), end-begin);
    return(buf);
}

__STATIC_INLINE void testITCM()
{
    __asm volatile (                                           
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        "  NOP                                         \n"
        :                      
        :                                  
        :);
}

__STATIC_INLINE void testMemAccess(const void *ptr)
{
    __asm volatile (
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        "  LDR r5,[%[src]] \n"
        :[src] "+r"(ptr)                  
        :
        :"r5");
}

void checkDTCMTime()
{
    printf("\r\nCheck DTCM time measurement (should be close to 50 cycles)\r\n");
    long mem;
    long a = time_in_cycles();
    testMemAccess(&mem);
    testMemAccess(&mem);
    testMemAccess(&mem);
    testMemAccess(&mem);
    testMemAccess(&mem);
    long b = time_in_cycles();
    printf("  Measured cycles %ld\r\n",(b-a)/5);
}

void checkExtMemTime(const void *ptr)
{
    printf("\r\nCheck external memory time measurement (should be close to 50 cycles with cache)\r\n");
    long a = time_in_cycles();
    testMemAccess(ptr);
    testMemAccess(ptr);
    testMemAccess(ptr);
    testMemAccess(ptr);
    testMemAccess(ptr);
    long b = time_in_cycles();
    printf("  Measured cycles %ld\r\n",(b-a)/5);
}

void checkITCMTime()
{
    printf("\r\nCheck ITCM time measurement (should be close to 25 cycles on board)\r\n");
    long a = time_in_cycles();
    testITCM();
    testITCM();
    testITCM();
    testITCM();
    testITCM();
    long b = time_in_cycles();
    printf("  Measured cycles %ld\r\n",(b-a)/5);
}

void demo() {

    int error = 0;
    int systick_status = 0;

    #if defined(MPS3)
    stdout_init();
    #endif

    SCB_EnableICache();
    SCB_EnableDCache();

#if defined(MPS3)
    GLCD_Initialize();
    //GLCD_WindowMax();
    GLCD_SetBackgroundColor(LightGrey);
    GLCD_SetForegroundColor(Black);
    GLCD_SetTextBackgroundColor(LightGrey);
    GLCD_SetTextForegroundColor(Black);
    GLCD_ClearScreen();
    GLCD_SetFontID(kFont16x24);
#endif

    const unsigned char *network_mem;
    const unsigned char *tokenizer_mem;

    // default parameters
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    const char *prompt = "Once upon a time";        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // parameter validation/overrides
    //if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    //
    if (rng_seed <= 0) rng_seed = (unsigned int)0;
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    #if !defined(MPS3)
      network_mem = load_mem((const char*)"net.bin",(const unsigned char*)0x70000000);
      tokenizer_mem = load_mem((const char*)"tok.bin",(const unsigned char*)0x71D00000);
    #else
      network_mem   = (const unsigned char*)0x70000000;
      tokenizer_mem = (const unsigned char*)0x71D00000;
    #endif                       

    if ((network_mem==nullptr) || (tokenizer_mem == nullptr))
    {
        printf("Data pointers null\r\n");
        return;
    }

    init_memory_area(&internal,(unsigned char*)internal_mem,NB_INT_MEM);

    printf("\r\nBuild transformer\r\n");

    // build the Transformer via the model .bin file
    Transformer transformer;

    uint32_t mem_stat = mem_usage;
    uint32_t int_mem_stat = internal.current_bytes;

    error = build_transformer(&transformer, network_mem);
    if (error != kNoError)
    {
        goto error;
    }
    if (steps == 0 || steps > MAX_SEQ_LEN) steps = MAX_SEQ_LEN; // override to ~max length

    printf("  Heap allocated transformer memory %u bytes (0x%X) \r\n",mem_usage-mem_stat,mem_usage-mem_stat);
    printf("  Internal allocated transformer memory %u bytes (0x%X) \r\n",internal.current_bytes-int_mem_stat,internal.current_bytes-int_mem_stat);


    printf("\r\nBuild tokenizer\r\n");
    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;

    mem_stat = mem_usage;
    int_mem_stat = internal.current_bytes;
    error = build_tokenizer(&tokenizer, tokenizer_mem, VOCAB_SIZE);
    if (error != kNoError)
    {
        goto error;
    }

    printf("  Heap allocated tokenizer memory %u bytes (0x%X) \r\n",mem_usage-mem_stat,mem_usage-mem_stat);
    printf("  Internal allocated tokenizer memory %u bytes (0x%X) \r\n",internal.current_bytes-int_mem_stat,internal.current_bytes-int_mem_stat);


    // build the Sampler
    printf("\r\nBuild sampler\r\n");
    mem_stat = mem_usage;
    int_mem_stat = internal.current_bytes;
    Sampler sampler;
    error = build_sampler(&sampler, VOCAB_SIZE, temperature, topp, rng_seed);
    if (error != kNoError)
    {
        goto error;
    }
    printf("  Heap allocated sampler memory %u bytes (0x%X) \r\n",mem_usage-mem_stat,mem_usage-mem_stat);
    printf("  Internal allocated sampler memory %u bytes (0x%X) \r\n",internal.current_bytes-int_mem_stat,internal.current_bytes-int_mem_stat);


    printf("\r\nTotal Heap allocated runtime memory %u bytes (0x%X) \r\n",mem_usage,mem_usage);
    printf("Total Internal allocated memory %u bytes (0x%X) \r\n",internal.current_bytes,internal.current_bytes);

    systick_status = SysTick_Config(MAX_SYSTICK);
    if (systick_status != 0)
    {
        printf("Can't enable SysTick\r\n");
    }

    checkDTCMTime();
    checkExtMemTime((void*)network_mem);
    checkITCMTime();

    printf("\r\nRunning ...\r\n\r\n");

    error = generate(&transformer, &tokenizer, &sampler, prompt, steps);
    if (error != kNoError)
    {
        goto error;
    }

    printf("\r\nEnd generation\r\n");

error:
   // memory and file handles cleanup
   free_sampler(&sampler);
   free_tokenizer(&tokenizer);
   free_transformer(&transformer);
   if (error!=0)
   {
      printf("An error occured %d. Allocated memory %u bytes (0x%X)\r\n",error,mem_usage,mem_usage);
   }  
}

