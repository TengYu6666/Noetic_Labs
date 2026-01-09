#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <cmath>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// 从CSV文件读取ImageNet类别映射
unordered_map<int, string> readImageNetLabelsFromCSV(const string& csv_path) {
    unordered_map<int, string> idx_to_labels;
    ifstream file(csv_path);

    if (!file.is_open()) {
        cerr << "错误: 无法打开CSV文件 " << csv_path << endl;
        return idx_to_labels;
    }

    string line;
    // 跳过表头
    getline(file, line);

    // 逐行读取
    while (getline(file, line)) {
        stringstream ss(line);
        string id_str, class_name;

        // 分割ID和类别名称
        if (getline(ss, id_str, ',') && getline(ss, class_name)) {
            try {
                int id = stoi(id_str);
                idx_to_labels[id] = class_name;
            }
            catch (const invalid_argument& e) {
                cerr << "警告: 无效的ID格式在CSV行: " << line << endl;
            }
            catch (const out_of_range& e) {
                cerr << "警告: ID超出范围在CSV行: " << line << endl;
            }
        }
    }

    file.close();
    cout << "成功从CSV读取 " << idx_to_labels.size() << " 个类别标签" << endl;
    return idx_to_labels;
}

// 自定义softmax函数将logits转换为概率
void softmax(const Mat& input, Mat& output) {
    Mat exp_output;
    double max_val;
    minMaxLoc(input, nullptr, &max_val);
    subtract(input, max_val, exp_output); // 减去最大值避免溢出
    exp(exp_output, exp_output);

    double sum_val = sum(exp_output)[0];
    exp_output /= sum_val;
    output = exp_output.clone();
}

int main() {
    // 设置控制台编码为UTF-8
    system("chcp 65001 > nul");

    // 路径设置
    string model_path = "";//模型地址
    string image_path = "";//测试图片地址
    string csv_path = "";//对照索引地址

    // 1. 加载CSV类别映射
    cout << "正在读取类别标签CSV文件..." << endl;
    unordered_map<int, string> labels = readImageNetLabelsFromCSV(csv_path);

    if (labels.empty()) {
        cerr << "警告: 未加载到任何类别标签!" << endl;
    }

    // 2. 加载ONNX模型
    cout << "正在加载ONNX模型..." << endl;
    Net net = readNetFromONNX(model_path);

    if (net.empty()) {
        cerr << "错误: 无法加载模型!" << endl;
        return -1;
    }
    cout << "模型加载成功!" << endl;

    // 3. 设置计算后端
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // 4. 读取输入图片
    cout << "正在加载输入图片..." << endl;
    Mat image = imread(image_path);

    if (image.empty()) {
        cerr << "错误: 无法加载图片!" << endl;
        return -1;
    }
    cout << "图片加载成功! 尺寸: " << image.size() << endl;

    // 5. 图像预处理
    cout << "正在预处理图像..." << endl;
    Mat blob;

    // ResNet标准预处理: 224x224, 归一化参数
    blobFromImage(image, blob, 1.0 / 255.0, Size(224, 224),
        Scalar(0.485, 0.456, 0.406), true, false);

    // 6. 执行推理
    cout << "正在执行模型推理..." << endl;
    net.setInput(blob);
    Mat output;
    net.forward(output);

    // 7. 处理输出结果
    cout << "正在处理推理结果..." << endl;

    // 将输出转换为概率
    Mat output_reshaped = output.reshape(1, 1);
    Mat probabilities;
    softmax(output_reshaped, probabilities);

    // 找到最高概率的类别
    Point class_id;
    double max_prob;
    minMaxLoc(probabilities, nullptr, &max_prob, nullptr, &class_id);

    // 8. 获取类别名称
    string class_name = "unknown";
    if (labels.count(class_id.x) > 0) {
        class_name = labels[class_id.x];
    }

    // 9. 输出结果
    cout << "\n=== 推理结果 ===" << endl;
    cout << "class_ID: " << class_id.x << endl;
    cout << "class_Name: " << class_name << endl;
    cout << "confidence: " << max_prob << endl;
    cout << "confidence per: " << max_prob * 100 << "%" << endl;
    cout << "================" << endl;

    return 0;
}

