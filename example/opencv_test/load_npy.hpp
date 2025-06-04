#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

cv::Mat load_npy_float32(const std::string& filename, int channels, int height, int width) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开 " << filename << std::endl;
        exit(-1);
    }

    // 跳过前面 npy header，简单粗暴跳过 128 字节（足够 cover 小型 header）
    file.seekg(128, std::ios::beg);

    // 计算元素数量
    int total = channels * height * width;
    std::vector<float> data(total);

    file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));

    cv::Mat mat(height, width, CV_32F, data.data());
    return mat.clone();  // clone 防止 data vector 被释放
}
