#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "load_npy.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
    // 加载 ONNX 模型
    Net net = readNetFromONNX("face_cnn.onnx");
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    // 加载一张图片 (例如第0张)
    Mat inputMat = load_npy_float32("test_data/test_images.npy", 1, 64, 64);

    // DNN blob (NCHW)
    Mat blob = blobFromImage(inputMat);

    // 设置输入名字
    net.setInput(blob, "input");

    // 推理
    Mat output = net.forward("output");

    // 输出预测结果
    Point classIdPoint;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int predicted_class = classIdPoint.x;

    cout << "预测类别: " << predicted_class << " 置信度: " << confidence << endl;

    return 0;
}
