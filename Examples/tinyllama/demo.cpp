/* Inference for Llama-2 Transformer model in pure C */

#include "RTE_Components.h"
#include  CMSIS_device_header

#include <iostream>
#include "common.h"
#include <cstdio>

#include <cstdio>
#include <cstdlib>
//#include <ctype>
#include <time.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>

#include "dsp/matrix_functions.h"
#include "dsp/support_functions.h"
#include "dsp/statistics_functions.h"
#include "dsp/basic_math_functions.h"


#include "dsp/matrix_functions_f16.h"
#include "dsp/support_functions_f16.h"
#include "dsp/statistics_functions_f16.h"
#include "dsp/basic_math_functions_f16.h"


#define DIM 288
#define HIDDEN_DIM 768
#define N_LAYERS 6
#define N_HEADS 6
#define N_KV_HEADS 6
#define VOCAB_SIZE 32000
#define MAX_SEQ_LEN 256
#define SHARED_WEIGHTS 1

#define KV_DIM ((DIM * N_KV_HEADS) / N_HEADS)


extern "C" {
    extern void demo();
}

// ----------------------------------------------------------------------------
// Transformer model

template<typename T>
struct TransformerWeights {
    // token embedding table
    T* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    T* rms_att_weight[N_LAYERS]; // (layer, dim) rmsnorm weights
    T* rms_ffn_weight[N_LAYERS]; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    T* wq[N_LAYERS]; // (layer, dim, n_heads * head_size)
    T* wk[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    T* wv[N_LAYERS]; // (layer, dim, n_kv_heads * head_size)
    T* wo[N_LAYERS]; // (layer, n_heads * head_size, dim)
    // weights for ffn
    T* w1[N_LAYERS]; // (layer, hidden_dim, dim)
    T* w2[N_LAYERS]; // (layer, dim, hidden_dim)
    T* w3[N_LAYERS]; // (layer, hidden_dim, dim)
    // final rmsnorm
    T* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    T* wcls;
} ;

template<typename T>
struct RunState {
    // current wave of activations
    T *x; // activation at current time stamp (dim,)
    T *xb; // same, but inside a residual branch (dim,)
    T *xb2; // an additional buffer just for convenience (dim,)
    T *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    T *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    T *q; // query (dim,)
    T *k; // key (dim,)
    T *v; // value (dim,)
    T *att; // buffer for scores/attention values (n_heads, seq_len)
    T *logits; // output logits
    // kv cache
    T* key_cache;   // (layer, seq_len, dim)
    T* value_cache; // (layer, seq_len, dim)
} ;

template<typename T>
struct Transformer {
    TransformerWeights<T> weights; // the weights of the model
    RunState<T> state; // buffers for the "wave" of activations in the forward pass
} ;

template<typename T>
void malloc_run_state(RunState<T>* s) {
    // we calloc instead of malloc to keep valgrind happy

    s->x = (T*)calloc(DIM, sizeof(T));
    s->xb = (T*)calloc(DIM, sizeof(T));
    s->xb2 = (T*)calloc(DIM, sizeof(T));
    s->hb = (T*)calloc(HIDDEN_DIM, sizeof(T));
    s->hb2 = (T*)calloc(HIDDEN_DIM, sizeof(T));
    s->q = (T*)calloc(DIM, sizeof(T));
    s->key_cache = (T*)calloc(N_LAYERS * MAX_SEQ_LEN * KV_DIM, sizeof(T));
    s->value_cache = (T*)calloc(N_LAYERS * MAX_SEQ_LEN * KV_DIM, sizeof(T));
    s->att = (T*)calloc(N_HEADS * MAX_SEQ_LEN, sizeof(T));
    s->logits = (T*)calloc(VOCAB_SIZE, sizeof(T));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stdout, "run state alloc error!\n");
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE);
    }
}

template<typename T>
void free_run_state(RunState<T>* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

template<typename T>
void memory_map_weights(TransformerWeights<T> *w, const unsigned char* ptr) {
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
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
    w->wcls = SHARED_WEIGHTS ? w->token_embedding_table : get_f16_tensor(ptr,ID++);
}

template<typename T>
void read_checkpoint(const unsigned char* memory, TransformerWeights<T>* weights) {
    //FILE *file = fopen(checkpoint, "rb");
    if (!memory) { 
        fprintf(stdout, "No memory for checkpoint\n"); 
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE); 
    }
    

    memory_map_weights<T>(weights, memory);
}

template<typename T>
void build_transformer(Transformer<T> *t, const unsigned char* memory) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint<T>(memory, &t->weights);
    // allocate the RunState buffers
    malloc_run_state<T>(&t->state);
}

template<typename T>
void free_transformer(Transformer<T>* t) {
    // free the RunState buffers
    free_run_state<T>(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float16_t* o, float16_t* x, float16_t* weight, int size) {

    // calculate sum of squares
    float16_t ss = 0.0f16;
    arm_power_f16(x, size, &ss);
    ss /= (_Float16)size;
    ss += 1e-5f16;
    ss = 1.0f16 / (_Float16)sqrtf(ss);

    // normalize and scale
    arm_scale_f16(x,ss,o,size);
    arm_mult_f16(weight,o,o,size);
    

}

void softmax(float16_t* x, int size) {

    // find max value (for numerical stability)
    float16_t max_val;
    arm_max_no_idx_f16(x, size, &max_val);

    // exp and sum
    float sum = 0.0f16;
    for (int i = 0; i < size; i++) {
        x[i] = (_Float16)expf((_Float16)x[i] - (_Float16)max_val);
        sum += (_Float16)x[i];
    }

    arm_scale_f16(x,1.0f16 / sum,x,size);

}



void matmul(float16_t* xout, float16_t* x, float16_t* w, int n, int d) {
    arm_matrix_instance_f16 W;
    W.numRows = d;
    W.numCols = n;
    W.pData = w;


   
    arm_mat_vec_mult_f16(&W,x,xout);

}


// CMSIS-DSP optimizations

__STATIC_INLINE float16_t dot_prod(const float16_t *a, 
                                   const float16_t *b, 
                                   uint32_t nb)
{
   float16_t result;

   arm_dot_prod_f16(a,b,nb,&result);

   return(result);
}



__STATIC_INLINE void vec_add(const float16_t *a, 
                              const float16_t *b, 
                              float16_t *o,
                              uint32_t nb)
{

   arm_add_f16(a,b,o,nb);

}


template<typename T>
struct Comp;

template<>
struct Comp<float16_t>
{
    using type = _Float16;
};

template<typename T>
T* forward(Transformer<T>* transformer, int token, int pos) {

    // a few convenience variables
    using C = typename Comp<T>::type;
    TransformerWeights<T>* w = &transformer->weights;
    RunState<T>* s = &transformer->state;
    T *x = s->x;
    int kv_mul = N_HEADS / N_KV_HEADS; // integer multiplier of the kv sharing in multiquery
    int head_size = DIM / N_HEADS;

    // copy the token embedding into x
    T* content_row = w->token_embedding_table + token * DIM;
    memcpy(x, content_row, DIM*sizeof(T));

    // forward all the layers
    for(int l = 0; l < N_LAYERS; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight[l], DIM);

        // key and value point to the kv cache
        int loff = l * MAX_SEQ_LEN * KV_DIM; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * KV_DIM;
        s->v = s->value_cache + loff + pos * KV_DIM;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq[l], DIM, DIM);
        matmul(s->k, s->xb, w->wk[l], DIM, KV_DIM);
        matmul(s->v, s->xb, w->wv[l], DIM, KV_DIM);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < DIM; i+=2) {
            int head_dim = i % head_size;
            T freq = (C)1.0f / (C)powf(10000.0f, head_dim / (T)head_size);
            T val = (C)pos * (C)freq;
            //T fcr = (T)cosf(val);
            //T fci = (T)sinf(val);
            T fcr = (T)arm_cos_f32((float32_t)val);
            T fci = (T)arm_sin_f32((float32_t)val);
            int rotn = i < KV_DIM ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                T* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                T v0 = vec[i];
                T v1 = vec[i+1];
                vec[i]   = (C)v0 * (C)fcr - (C)v1 * (C)fci;
                vec[i+1] = (C)v0 * (C)fci + (C)v1 * (C)fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < N_HEADS; h++) {
            // get the query vector for this head
            T* q = s->q + h * head_size;
            // attention scores for this head
            T* att = s->att + h * MAX_SEQ_LEN;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                T* k = s->key_cache + loff + t * KV_DIM + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                T score = 0.0f;
                score = dot_prod(q,k,head_size);
                /*
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                */
                score /= (C)sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            T* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(T));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                T* v = s->value_cache + loff + t * KV_DIM + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                T a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += (C)a * (C)v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo[l], DIM, DIM);

        // residual connection back into x
        vec_add(x,s->xb2,x,DIM);
        /*
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        */

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight[l], DIM);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1[l], DIM, HIDDEN_DIM);
        matmul(s->hb2, s->xb, w->w3[l], DIM, HIDDEN_DIM);

        // SwiGLU non-linearity
        for (int i = 0; i < HIDDEN_DIM; i++) {
            T val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (C)((C)1.0f / (C)((C)1.0f + (C)expf(-val)));
            // elementwise multiply with w3(x)
            val *= (C)(s->hb2[i]);
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2[l], HIDDEN_DIM, DIM);

        // residual connection

        vec_add(x,s->xb,x,DIM);
        /*
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
        */
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, DIM);

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
    memcpy(&v,p,4);

    return(*reinterpret_cast<const T*>(&v));
}
void build_tokenizer(Tokenizer* t, const unsigned char* memory, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    if (!t->vocab || !t->vocab_scores)
    {
        fprintf(stdout, "build tokenizer malloc failed!\n");
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE);
    }
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    if (!memory) { 
        fprintf(stdout, "no memory for tokenizer\n"); 
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE); 
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

        t->vocab[i] = (char *)malloc(len + 1);
        if (!t->vocab[i])
        {
            fprintf(stdout, "malloc failed for vocab %d!\n",i);
            #if defined(MPS3)
               while(1);
            #endif
            exit(EXIT_FAILURE);
        }
        memcpy(t->vocab[i],memory,len);
        memory += len;

        t->vocab[i][len] = '\0'; // add the string terminating token
    }
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
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
    printf("%s", piece);
}

int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str, .id = 0 }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { 
        fprintf(stdout, "cannot encode NULL text\n"); 
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE); 
    }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        if (!t->sorted_vocab) 
        { 
            fprintf(stdout, "sorted vocab alloc error\n"); 
            #if defined(MPS3)
               while(1);
            #endif
            exit(EXIT_FAILURE); 
        }
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    if (!str_buffer) 
    { 
            fprintf(stdout, "str buffer alloc error\n"); 
            #if defined(MPS3)
               while(1);
            #endif
            exit(EXIT_FAILURE); 
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

    free(str_buffer);
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

template<typename T>
int sample_argmax(T* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    T max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

template<typename T>
int sample_mult(T* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    T cdf = 0.0f;
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

template<typename T>
int sample_topp(T* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const T cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    T cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    T r = coin * cumulative_prob;
    T cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
    if (!sampler->probindex) 
    { 
        fprintf(stdout, "sampler alloc error\n"); 
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE); 
    }
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
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

template<typename T>
int sample(Sampler* sampler, T* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
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

#define INIT_SYSTICK \
 SysTick->CTRL=0;         \
 SysTick->LOAD=0xFFFFFFUL;\
 SysTick->VAL=0;          \
 SysTick->CTRL=5;         \
 while (SysTick->VAL==0)\
    ; 
    



extern "C" uint32_t SystemCoreClock;

long time_in_ms() {
    

    return(1000*SysTick->VAL/SystemCoreClock);
}

// ----------------------------------------------------------------------------
// generation loop

template<typename T>
void generate(Transformer<T> *transformer, Tokenizer *tokenizer, Sampler *sampler, const char *prompt, int steps) {
    const char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    if (!prompt_tokens) 
    { 
        fprintf(stdout, "prompt token alloc error\n"); 
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE); 
    }

    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stdout, "something is wrong, expected at least 1 prompt token\n");
        #if defined(MPS3)
           while(1);
        #endif
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        T* logits = forward(transformer, token, pos);
        #if defined(STATS)
        printf("Max vec = %lu\r\n",maxvec);
        #endif

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
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}




static const std::string s[3]={"test","network.dat"};

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

    printf("%s : %lld bytes,(%llX)\r\n",checkpoint,end-begin,end-begin);
    const unsigned char *buf;
    if (refbuf!=nullptr)
    {
       buf = refbuf;
    }
    else
    {
       buf =  (const unsigned char*)malloc(end-begin);
    }
    if (buf == nullptr)
    {
        printf("Not enough memory\r\n");
        return(nullptr);
    }

    fin.read(const_cast<char*>(reinterpret_cast<const char*>(buf)), end-begin);
    return(buf);
}


void demo() {

    #if defined(MPS3)
    stdout_init();
    #endif

    #if defined(MPS3)
    printf("\r\nStart new test\r\n");
    #endif

    const unsigned char *network_mem;
    const unsigned char *tokenizer_mem;

    // default parameters
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    const char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // parameter validation/overrides
    //if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    //
    if (rng_seed <= 0) rng_seed = (unsigned int)0;
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    #if defined(MPS3)
    printf("\r\nLoad data\r\n");
    #endif
    #if !defined(MPS3)
      network_mem = load_mem((const char*)"network.dat",(const unsigned char*)0x70000000);
      tokenizer_mem = load_mem((const char*)"tokenizer.bin",(const unsigned char*)0x71D00000);
    #else
      network_mem   = (const unsigned char*)0x70000000;
      tokenizer_mem = (const unsigned char*)0x71D00000;
    #endif                       

    if ((network_mem==nullptr) || (tokenizer_mem == nullptr))
    {
        printf("Data pointers null\r\n");
        return;
    }

    printf("\r\nBuild transformer\r\n");

    // build the Transformer via the model .bin file
    Transformer<float16_t> transformer;

    build_transformer(&transformer, network_mem);
    if (steps == 0 || steps > MAX_SEQ_LEN) steps = MAX_SEQ_LEN; // override to ~max length


    printf("\r\nBuild tokenizer\r\n");
    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;

    build_tokenizer(&tokenizer, tokenizer_mem, VOCAB_SIZE);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, VOCAB_SIZE, temperature, topp, rng_seed);

    printf("\r\nRun\r\n");

    INIT_SYSTICK;

    // run!
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
   

    printf("\r\nEnd generation\r\n");

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    //#if !defined(MPS3)
    //free((void*)network_mem);
    //free((void*)tokenizer_mem);
    //#endif
    
}

