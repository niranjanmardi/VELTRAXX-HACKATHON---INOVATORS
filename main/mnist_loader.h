#pragma once
#include <vector>
#include <string>

struct MNISTData {
    std::vector<float> images;   
    std::vector<int>   labels;   
    int num_images;
};

bool load_mnist_test(const std::string& images_path,
                     const std::string& labels_path,
                     MNISTData& data);
