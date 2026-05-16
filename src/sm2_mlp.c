// sm2_mlp.c - Feed-Forward Network (MLP / Swiglu)
// 
// SmolLM2 uses SwiGLU activation:
//   f(x) = Swish(x) * GELU(gate(x))
// where:
//   gate = x @ W_gate
//   up = x @ W_up
//   down = (Swish(gate) * up) @ W_down

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "smollm2.h"

#define SM2_M_SQRT2_M_PI 2.5066282746310005024157652848110f  // sqrt(2/pi)

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

// GELU (Gaussian Error Linear Unit) approximation
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static float gelu(float x) {
    float c = SM2_M_SQRT2_M_PI;
    float t = tanhf(c * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

// Swish activation: x * sigmoid(x)
static float swish(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// FFN FORWARD
// ============================================================================

void sm2_ffn_forward(float* out, const float* input, 
                     const float* gate_proj_w, const float* gate_proj_b,
                     const float* up_proj_w, const float* up_proj_b,
                     const float* down_proj_w, const float* down_proj_b,
                     int input_dim, int hidden_dim, int output_dim) {
    
    // gate = GELU(input @ W_gate + b_gate)
    float* gate = (float*)malloc(hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        float sum = gate_proj_b ? gate_proj_b[i] : 0.0f;
        for (int j = 0; j < input_dim; j++) {
            sum += input[j] * gate_proj_w[i * input_dim + j];
        }
        gate[i] = gelu(sum);
    }
    
    // up = input @ W_up + b_up
    float* up = (float*)malloc(hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        float sum = up_proj_b ? up_proj_b[i] : 0.0f;
        for (int j = 0; j < input_dim; j++) {
            sum += input[j] * up_proj_w[i * input_dim + j];
        }
        up[i] = sum; // No activation on up projection
    }
    
    // ffn = swish(gate) * up (SwiGLU)
    float* ffn = (float*)malloc(hidden_dim * sizeof(float));
    for (int i = 0; i < hidden_dim; i++) {
        ffn[i] = swish(gate[i]) * up[i];
    }
    
    // out = ffn @ W_down + b_down
    for (int i = 0; i < output_dim; i++) {
        float sum = down_proj_b ? down_proj_b[i] : 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            sum += ffn[j] * down_proj_w[i * hidden_dim + j];
        }
        out[i] = sum;
    }
    
    free(gate);
    free(up);
    free(ffn);
}

// In-place FFN for preallocated buffer usage
void sm2_ffn_forward_inplace(float* io, 
                             const float* gate_proj_w, const float* gate_proj_b,
                             const float* up_proj_w, const float* up_proj_b,
                             const float* down_proj_w, const float* down_proj_b,
                             int input_dim, int hidden_dim, int output_dim,
                             float* scratch) {
    
    // Use scratch buffer for intermediate results
    float* gate = scratch;
    float* up = scratch + hidden_dim;
    float* ffn = scratch + hidden_dim * 2;
    
    // gate = GELU(io @ W_gate + b_gate)
    for (int i = 0; i < hidden_dim; i++) {
        float sum = gate_proj_b ? gate_proj_b[i] : 0.0f;
        for (int j = 0; j < input_dim; j++) {
            sum += io[j] * gate_proj_w[i * input_dim + j];
        }
        gate[i] = gelu(sum);
    }
    
    // up = io @ W_up + b_up
    for (int i = 0; i < hidden_dim; i++) {
        float sum = up_proj_b ? up_proj_b[i] : 0.0f;
        for (int j = 0; j < input_dim; j++) {
            sum += io[j] * up_proj_w[i * input_dim + j];
        }
        up[i] = sum;
    }
    
    // ffn = swish(gate) * up
    for (int i = 0; i < hidden_dim; i++) {
        ffn[i] = swish(gate[i]) * up[i];
    }
    
    // io = ffn @ W_down + b_down
    for (int i = 0; i < output_dim; i++) {
        float sum = down_proj_b ? down_proj_b[i] : 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            sum += ffn[j] * down_proj_w[i * hidden_dim + j];
        }
        io[i] = sum;
    }
}