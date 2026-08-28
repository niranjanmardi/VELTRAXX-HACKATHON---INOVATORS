#pragma once
#include <vector>
#include <cstdint>
#include <string>

struct CSRLayer {
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t nnz;
    std::vector<float>   values;       
    std::vector<int32_t> col_indices;  
    std::vector<int32_t> row_ptr;      
    std::vector<float>   bias;         
};

struct SparseModel {
    std::vector<CSRLayer> layers;
};

bool load_sparse_model(const std::string& filepath, SparseModel& model);
void print_model_info(const SparseModel& model);
size_t get_sparse_size_bytes(const SparseModel& model);
size_t get_dense_size_bytes(const SparseModel& model);
