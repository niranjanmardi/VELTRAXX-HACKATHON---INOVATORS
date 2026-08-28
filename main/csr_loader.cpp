#include "csr_loader.h"
#include <iostream>
#include <fstream>
#include <iomanip>

bool load_sparse_model(const std::string& filepath, SparseModel& model) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << std::endl;
        return false;
    }

    char magic[4];
    if (!file.read(magic, 4) || magic[0] != 'V' || magic[1] != 'C' || magic[2] != 'S' || magic[3] != 'R') {
        std::cerr << "Error: Invalid magic, not a VCSR file." << std::endl;
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "Error: Unsupported version " << version << std::endl;
        return false;
    }

    uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    model.layers.resize(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
        auto& layer = model.layers[i];
        file.read(reinterpret_cast<char*>(&layer.num_rows), sizeof(layer.num_rows));
        file.read(reinterpret_cast<char*>(&layer.num_cols), sizeof(layer.num_cols));
        file.read(reinterpret_cast<char*>(&layer.nnz), sizeof(layer.nnz));

        layer.values.resize(layer.nnz);
        file.read(reinterpret_cast<char*>(layer.values.data()), layer.nnz * sizeof(float));

        layer.col_indices.resize(layer.nnz);
        file.read(reinterpret_cast<char*>(layer.col_indices.data()), layer.nnz * sizeof(int32_t));

        layer.row_ptr.resize(layer.num_rows + 1);
        file.read(reinterpret_cast<char*>(layer.row_ptr.data()), (layer.num_rows + 1) * sizeof(int32_t));

        layer.bias.resize(layer.num_rows);
        file.read(reinterpret_cast<char*>(layer.bias.data()), layer.num_rows * sizeof(float));
    }

    return true;
}

void print_model_info(const SparseModel& model) {
    std::cout << "Model contains " << model.layers.size() << " layers.\n";
    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        double sparsity = 100.0 * (1.0 - static_cast<double>(layer.nnz) / (layer.num_rows * layer.num_cols));
        std::cout << "Layer " << i << ": shape (" << layer.num_rows << "x" << layer.num_cols << "), "
                  << "nnz = " << layer.nnz << ", sparsity = " << std::fixed << std::setprecision(2) << sparsity << "%\n";
    }
}

size_t get_sparse_size_bytes(const SparseModel& model) {
    size_t total = 0;
    for (const auto& layer : model.layers) {
        total += layer.nnz * sizeof(float);
        total += layer.nnz * sizeof(int32_t);
        total += (layer.num_rows + 1) * sizeof(int32_t);
        total += layer.num_rows * sizeof(float);
    }
    return total;
}

size_t get_dense_size_bytes(const SparseModel& model) {
    size_t total = 0;
    for (const auto& layer : model.layers) {
        total += layer.num_rows * layer.num_cols * sizeof(float);
        total += layer.num_rows * sizeof(float);
    }
    return total;
}
