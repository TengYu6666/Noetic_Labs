# OpenCV ONNX 推理项目

这是一个使用 OpenCV DNN 模块进行 ONNX 模型推理的 C++ 项目，主要用于图像分类任务。

## 功能特性

- 使用 OpenCV DNN 模块加载和推理 ONNX 模型
- 支持 ResNet18 等 ImageNet 预训练模型
- 从 CSV 文件读取类别名称映射
- 支持中文类别名称输出
- 跨平台支持（通过 CMake）

## 技术栈

- C++11/14
- OpenCV 12
- ONNX Runtime (可选)
- CMake 3.16+

## 安装步骤

### 1. 安装依赖

#### Windows

1. 安装 [OpenCV](https://opencv.org/releases/)
2. 安装 [CMake](https://cmake.org/download/)
3. 安装 [Visual Studio](https://visualstudio.microsoft.com/zh-hans/downloads/) (推荐 2019 或更高版本)

#### Linux

```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev cmake g++
```

### 2. 克隆项目

```bash
git clone <repository-url>
cd <repository-name>
```

### 3. 构建项目

#### 使用 CMake (推荐)

```bash
mkdir build
cd build
cmake ..
make
```

#### 使用 Visual Studio

1. 打开 `tuilimoxing.slnx` 或 `tuilimoxing.vcxproj`
2. 选择编译配置 (Debug/Release)
3. 点击 "生成" 按钮

## 使用方法

### 准备文件

- ONNX 模型文件 (如：`resnet18_imagenet.onnx`)
- 测试图像文件 (如：`banana.jpg`)
- 类别索引 CSV 文件 (如：`imagenet_class_index.csv`)

### 运行程序

```bash
./tuilimoxing <model-path> <image-path> <csv-path>
```

示例：

```bash
./tuilimoxing resnet18_imagenet.onnx banana1.jpg imagenet_class_index.csv
```

## 项目结构

```
tuilimoxing/
├── README.md            # 项目说明文档
├── tuilimoxing/         # 主项目目录
│   ├── main.cpp        # 主要源代码
├── imagenet_class_index.csv  # ImageNet 类别索引
├── resnet18_imagenet.onnx    # ONNX 模型文件
└── banana1.jpg            # 测试图像

```

