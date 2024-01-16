#include "sd.h"
#include "rng.hpp"

#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include "ggml/ggml.h"

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <set>

#ifdef SD_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

// graph sizes
#define SD_CLIP_GRAPH_SIZE      2048
#define SD_UNET_GRAPH_SIZE      8096
#define SD_VAE_GRAPH_SIZE       2048
#define SD_LORA_GRAPH_SIZE      10240

// primitive network blocks
struct sample_block {
    struct ggml_tensor* conv_w;
    struct ggml_tensor* conv_b;
};

struct resnet_block {
    struct ggml_tensor* norm1_w;
    struct ggml_tensor* norm1_b;

    struct ggml_tensor* conv1_w;
    struct ggml_tensor* conv1_b;

    struct ggml_tensor* norm2_w;
    struct ggml_tensor* norm2_b;

    struct ggml_tensor* conv2_w;
    struct ggml_tensor* conv2_b;

    struct ggml_tensor* skip_w;
    struct ggml_tensor* skip_b;
};

enum attn_block_type {
    ATTN_MLP, // clip
    ATTN_CROSS_FFN, // spatial transformer
    ATTN_NORMAL // vae
};

struct attention_block {
    attn_block_type type;

    struct ggml_tensor* norm_w;
    struct ggml_tensor* norm_b;

    // qkv weights (self)
    struct ggml_tensor* q_w;
    struct ggml_tensor* k_w;
    struct ggml_tensor* v_w;
    struct ggml_tensor* kqv_out_w;
    struct ggml_tensor* kqv_out_b;

    // qkv weights (cross)
    struct ggml_tensor* cq_w;
    struct ggml_tensor* ck_w;
    struct ggml_tensor* cv_w;
    struct ggml_tensor* ckqv_out_w;
    struct ggml_tensor* ckqv_out_b;

    // qkv bias
    struct ggml_tensor* q_b;
    struct ggml_tensor* k_b;
    struct ggml_tensor* v_b;

    struct ggml_tensor* norm2_w;
    struct ggml_tensor* norm2_b;

    struct ggml_tensor* norm3_w;
    struct ggml_tensor* norm3_b;

    // ff
    struct ggml_tensor* proj_out_w;
    struct ggml_tensor* proj_out_b;

    struct ggml_tensor* ff_2_w;
    struct ggml_tensor* ff_2_b;
};

struct spatial_transformer {
    
};

struct tiny_vae_block {

};

struct esrgan_block {

};

struct control_encoder {

};

// unet complex network layers
struct sd_unet_encoder {

};

struct sd_unet_decoder {

};

// VAE complex network layers
struct vae_encoder {

};

struct vae_decoder {

};

struct autoencoder_kl {

};

// TinyAutoEncoder complex network layers
struct tiny_vae_encoder {

};

struct tiny_vae_decoder {

};

// conditioners
enum clip_version {
    CLIP_SD1,
    CLIP_SD2,
    CLIP_SDXL
};

struct clip_mparams {
    clip_version version            = CLIP_SD1;
    // network hparams
    int32_t vocab_size              = 49408;
    int32_t max_position_embeddings = 77;
    int32_t hidden_size             = 768;   // 1024 for OPEN_CLIP_VIT_H_14
    int32_t intermediate_size       = 3072;  // 4096 for OPEN_CLIP_VIT_H_14
    int32_t n_head                  = 12;    // num_attention_heads, 16 for OPEN_CLIP_VIT_H_14
    int32_t num_hidden_layers       = 12;    // 24 for OPEN_CLIP_VIT_H_14
    int32_t layer_idx               = 11;
    int32_t projection_dim          = 1280;  // only for OPEN_CLIP_VIT_BIGG_14
    bool with_final_ln              = true;
};

struct clip_model {
    struct ggml_tensor* position_ids;
    struct ggml_tensor* token_embed_weight;
    struct ggml_tensor* position_embed_weight;

    attention_block layers[13];

    struct ggml_tensor* final_ln_w;
    struct ggml_tensor* final_ln_b;
};

struct clip_tokenizer {

};

// control model
struct controlnet_model {
    sd_unet_encoder encoder;

};

// upscaler model
struct ersgan_model {

};

struct sd_model {
    sd_unet_encoder encoder;
    sd_unet_decoder decoder;
};


