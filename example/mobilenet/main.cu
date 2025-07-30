#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

vector<string> readClassNames(const string& filename) {
    vector<string> classes;
    ifstream ifs(filename);
    if (!ifs.is_open()) {
        cerr << "cannot open label file: " << filename << endl;
        exit(-1);
    }
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }
    return classes;
}

int main() {
    // find more pre-trained models at: https://github.com/onnx/models/
    string modelPath = "mobilenetv2-10.onnx"; 
    Net net = readNetFromONNX(modelPath);
    if (net.empty()) {
        cerr << "cannot load model: " << modelPath << endl;
        return -1;
    }
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    vector<string> classes = readClassNames("imagenet_classes.txt");

    string imagePath = "tree_frog.jpg"; 
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "cannot load image: " << imagePath << endl;
        return -1;
    }

    Mat blob;
    blobFromImage(image, blob, 1.0/255.0, Size(224, 224), Scalar(0.485, 0.456, 0.406), true, false);
    
    Scalar mean(0.485, 0.456, 0.406);
    Scalar std(0.229, 0.224, 0.225);
    divide(blob - mean, std, blob);

    net.setInput(blob);
    Mat output = net.forward();

    Point classIdPoint;
    double confidence;
    minMaxLoc(output.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);

    // int classId = classIdPoint.x;
    // string label = format("%s: %.2f", classes[classId].c_str(), confidence);
    // putText(image, label, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    // imshow("Result", image);
    // waitKey(0);
    
    cout << "Prediction: " << classes[classIdPoint.x] 
         << " (Confidence: " << fixed << setprecision(2) << confidence * 100 << "%)" 
         << endl;

    return 0;
}