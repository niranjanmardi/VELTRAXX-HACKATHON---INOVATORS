#pragma once
#include "csr_loader.h"

int sparse_forward(const SparseModel& model,
                   const float* input, int input_dim,
                   float* output, int output_dim);

void relu_inplace(float* data, int size);
void softmax_inplace(float* data, int size);

void sparse_matmul(const CSRLayer& layer,
                   const float* input,
                   float* output);
