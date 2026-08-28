#include "mnist_loader.h"
#include <fstream>
#include <iostream>
#include <cstdint>

inline uint32_t swap_endian(uint32_t val) {
    return ((val & 0xFF000000) >> 24) | ((val & 0x00FF0000) >> 8) |
           ((val & 0x0000FF00) << 8)  | ((val & 0x000000FF) << 24);
}

bool load_mnist_test(const std::string& images_path,
                     const std::string& labels_path,
                     MNISTData& data) {
    std::ifstream img_file(images_path, std::ios::binary);
    std::ifstream lbl_file(labels_path, std::ios::binary);

    if (!img_file.is_open() || !lbl_file.is_open()) {
        std::cerr << "Error: Could not open MNIST files." << std::endl;
        return false;
    }

    uint32_t magic, num_images, num_rows, num_cols;
    img_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) return false;

    img_file.read(reinterpret_cast<char*>(&num_images), 4);
    num_images = swap_endian(num_images);

    img_file.read(reinterpret_cast<char*>(&num_rows), 4);
    num_rows = swap_endian(num_rows);

    img_file.read(reinterpret_cast<char*>(&num_cols), 4);
    num_cols = swap_endian(num_cols);

    uint32_t l_magic, l_num;
    lbl_file.read(reinterpret_cast<char*>(&l_magic), 4);
    l_magic = swap_endian(l_magic);
    if (l_magic != 2049) return false;

    lbl_file.read(reinterpret_cast<char*>(&l_num), 4);
    l_num = swap_endian(l_num);

    if (num_images != l_num) return false;
    data.num_images = num_images;

    int image_size = num_rows * num_cols;
    data.images.resize(num_images * image_size);
    std::vector<uint8_t> raw_images(num_images * image_size);
    img_file.read(reinterpret_cast<char*>(raw_images.data()), num_images * image_size);

    for (size_t i = 0; i < raw_images.size(); ++i) {
        data.images[i] = raw_images[i] / 255.0f;
    }

    std::vector<uint8_t> raw_labels(num_images);
    lbl_file.read(reinterpret_cast<char*>(raw_labels.data()), num_images);
    data.labels.assign(raw_labels.begin(), raw_labels.end());

    return true;
}
