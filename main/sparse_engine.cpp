#include "sparse_engine.h"
#include <cmath>
#include <algorithm>

void sparse_matmul(const CSRLayer& layer, const float* input, float* output) {
    for (uint32_t i = 0; i < layer.num_rows; ++i) {
        float sum = layer.bias[i];
        int32_t start = layer.row_ptr[i];
        int32_t end = layer.row_ptr[i + 1];
        for (int32_t j = start; j < end; ++j) {
            sum += layer.values[j] * input[layer.col_indices[j]];
        }
        output[i] = sum;
    }
}

void relu_inplace(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void softmax_inplace(float* data, int size) {
    if (size == 0) return;
    float max_val = data[0];
    for (int i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    
    for (int i = 0; i < size; ++i) {
        data[i] /= sum;
    }
}

int sparse_forward(const SparseModel& model,
                   const float* input, int input_dim,
                   float* output, int output_dim) {
    if (model.layers.empty()) return -1;
    
    std::vector<float> buffer1(input, input + input_dim);
    std::vector<float> buffer2;
    
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        buffer2.resize(layer.num_rows);
        
        sparse_matmul(layer, buffer1.data(), buffer2.data());
        
        if (i < model.layers.size() - 1) {
            relu_inplace(buffer2.data(), layer.num_rows);
        } else {
            softmax_inplace(buffer2.data(), layer.num_rows);
        }
        
        buffer1 = buffer2;
    }
    
    int max_idx = 0;
    float max_val = buffer1[0];
    for (size_t i = 0; i < buffer1.size(); ++i) {
        if (i < (size_t)output_dim) {
            output[i] = buffer1[i];
        }
        if (buffer1[i] > max_val) {
            max_val = buffer1[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}
