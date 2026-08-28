#include "benchmark.h"
#include "sparse_engine.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

BenchmarkResult run_benchmark(
    const SparseModel& model,
    const float* samples,      
    const int* labels,         
    int num_samples,
    int input_dim,
    int output_dim) {
    
    BenchmarkResult res{};
    res.num_samples = num_samples;
    res.sparse_size_bytes = get_sparse_size_bytes(model);
    res.dense_size_bytes = get_dense_size_bytes(model);
    res.memory_reduction = static_cast<double>(res.dense_size_bytes) / static_cast<double>(res.sparse_size_bytes);
    res.correct = 0;
    
    std::vector<float> output(output_dim);
    double total_time_ms = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        const float* input = samples + i * input_dim;
        
        auto start = std::chrono::high_resolution_clock::now();
        int pred = sparse_forward(model, input, input_dim, output.data(), output_dim);
        auto end = std::chrono::high_resolution_clock::now();
        
        total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
        
        if (pred == labels[i]) {
            res.correct++;
        }
    }
    
    res.total_ms = total_time_ms;
    res.avg_inference_ms = total_time_ms / num_samples;
    res.accuracy = 100.0 * res.correct / num_samples;
    
    return res;
}

void print_benchmark_result(const BenchmarkResult& result) {
    std::cout << "\n=== C++ Sparse Engine Timing Report ===\n";
    std::cout << "Samples tested:      " << result.num_samples << "\n";
    std::cout << "Total time:          " << std::fixed << std::setprecision(2) << result.total_ms << " ms\n";
    std::cout << "Avg per sample:      " << std::fixed << std::setprecision(3) << result.avg_inference_ms << " ms\n";
    std::cout << "Accuracy:            " << std::fixed << std::setprecision(2) << result.accuracy << "%\n";
    std::cout << "Status: " << (result.accuracy >= 90.0 ? "PASS" : "FAIL") << "\n";
}

void save_benchmark_result(const BenchmarkResult& result, const std::string& filepath) {
    std::filesystem::path p(filepath);
    std::filesystem::create_directories(p.parent_path());
    
    std::ofstream out(filepath);
    if (out.is_open()) {
        out << "=== C++ Sparse Engine Timing Report ===\n";
        out << "Samples tested:      " << result.num_samples << "\n";
        out << "Total time:          " << std::fixed << std::setprecision(2) << result.total_ms << " ms\n";
        out << "Avg per sample:      " << std::fixed << std::setprecision(3) << result.avg_inference_ms << " ms\n";
        out << "Accuracy:            " << std::fixed << std::setprecision(2) << result.accuracy << "%\n";
        out << "Status: " << (result.accuracy >= 90.0 ? "PASS" : "FAIL") << "\n";
    }
}
