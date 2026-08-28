#pragma once
#include "csr_loader.h"
#include <string>

struct BenchmarkResult {
    double avg_inference_ms;    
    double total_ms;            
    int num_samples;            
    size_t sparse_size_bytes;   
    size_t dense_size_bytes;    
    double memory_reduction;    
    int correct;                
    double accuracy;            
};

BenchmarkResult run_benchmark(
    const SparseModel& model,
    const float* samples,      
    const int* labels,         
    int num_samples,
    int input_dim,
    int output_dim
);

void print_benchmark_result(const BenchmarkResult& result);
void save_benchmark_result(const BenchmarkResult& result, const std::string& filepath);
