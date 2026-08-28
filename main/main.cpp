#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "csr_loader.h"
#include "sparse_engine.h"
#include "benchmark.h"
#include "mnist_loader.h"

int main(int argc, char** argv) {
    std::string model_path = "outputs/sparse_weights.bin";
    std::string images_path = "data/MNIST/raw/t10k-images-idx3-ubyte";
    std::string labels_path = "data/MNIST/raw/t10k-labels-idx1-ubyte";
    int num_samples = 1000;
    std::string outdir = "logs/";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--images" && i + 1 < argc) images_path = argv[++i];
        else if (arg == "--labels" && i + 1 < argc) labels_path = argv[++i];
        else if (arg == "--samples" && i + 1 < argc) num_samples = std::stoi(argv[++i]);
        else if (arg == "--outdir" && i + 1 < argc) outdir = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: ./sparse_engine [options]\n"
                      << "  --model   path  Path to sparse_weights.bin (default: outputs/sparse_weights.bin)\n"
                      << "  --images  path  Path to MNIST test images (default: data/MNIST/raw/t10k-images-idx3-ubyte)\n"
                      << "  --labels  path  Path to MNIST test labels (default: data/MNIST/raw/t10k-labels-idx1-ubyte)\n"
                      << "  --samples N     Number of test samples to run (default: 1000)\n"
                      << "  --outdir  path  Output directory for reports (default: logs/)\n"
                      << "  --help          Show this help\n";
            return 0;
        }
    }

    std::cout << "=== VELTRAXX Sparse Inference Engine ===\n";

    SparseModel model;
    if (!load_sparse_model(model_path, model)) {
        return 1;
    }

    print_model_info(model);

    size_t sparse_size = get_sparse_size_bytes(model);
    size_t dense_size = get_dense_size_bytes(model);
    double ratio = static_cast<double>(dense_size) / sparse_size;

    std::cout << "Dense equivalent size:  " << dense_size << " bytes (" << (dense_size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "Sparse CSR model size:  " << sparse_size << " bytes (" << (sparse_size / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "Memory reduction ratio: " << std::fixed << std::setprecision(2) << ratio << "x\n";

    std::filesystem::create_directories(outdir);
    std::ofstream mem_out(outdir + "/cpp_memory_report.txt");
    if (mem_out.is_open()) {
        mem_out << "=== C++ Sparse Engine Memory Report ===\n";
        mem_out << "Dense equivalent size:  " << dense_size << " bytes (" << std::fixed << std::setprecision(2) << (dense_size / 1024.0 / 1024.0) << " MB)\n";
        mem_out << "Sparse CSR model size:  " << sparse_size << " bytes (" << std::fixed << std::setprecision(2) << (sparse_size / 1024.0 / 1024.0) << " MB)\n";
        mem_out << "Memory reduction ratio: " << std::fixed << std::setprecision(2) << ratio << "x\n";
        mem_out << "Status: " << (ratio >= 2.0 ? "PASS (target >=2.0x)" : "FAIL") << "\n";
    }

    MNISTData data;
    if (!load_mnist_test(images_path, labels_path, data)) {
        return 1;
    }

    num_samples = std::min(num_samples, data.num_images);
    int input_dim = 784; 
    int output_dim = model.layers.back().num_rows;

    BenchmarkResult res = run_benchmark(model, data.images.data(), data.labels.data(), num_samples, input_dim, output_dim);
    
    print_benchmark_result(res);
    save_benchmark_result(res, outdir + "/cpp_timing_report.txt");

    return 0;
}
