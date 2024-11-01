# SPEED SAM C++ TENSORRT
![SAM C++ TENSORRT](assets/speed_sam_cpp_tenosrrt.PNG)

<a href="https://github.com/hamdiboukamcha/SPEED-SAM-C-TENSORRT" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>

  <a href="https://github.com/hamdiboukamcha/SPEED-SAM-C-TENSORRT?tab=GPL-3.0-1-ov-file" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>

## 🌐 Overview
A high-performance C++ implementation for SAM (segment anything model) using TensorRT and CUDA, optimized for real-time image segmentation tasks.

## 📢 Updates
    Model Conversion: Build TensorRT engines from ONNX models for accelerated inference.
    Segmentation with Points and BBoxes: Easily segment images using selected points or bounding boxes.
    FP16 Precision: Choose between FP16 and FP32 for speed and precision balance.
    Dynamic Shape Support: Efficient handling of variable input sizes using optimization profiles.
    CUDA Optimization: Leverage CUDA for preprocessing and efficient memory handling.

## 📢 Performance 
 ### Infernce Time 

| Component                  | SpeedSAM |
|----------------------------|-----------|
| **Image Encoder**          |           |
| Parameters                  | 5M        |
| Speed                       | 8ms       |
| **Mask Decoder**           |           |
| Parameters                  | 3.876M    |
| Speed                       | 4ms       |
| **Whole Pipeline (Enc+Dec)** |         |
| Parameters                  | 9.66M     |
| Speed                       | 12ms      |
### Results
![SPEED-SAM-C-TENSORRT RESULT](assets/Speed_SAM_Results.JPG)

## 📂 Project Structure
    SPEED-SAM-CPP-TENSORRT/
    ├── include
    │   ├── config.h          # Model configuration and macros
    │   ├── cuda_utils.h      # CUDA utility macros
    │   ├── engineTRT.h       # TensorRT engine management
    │   ├── logging.h         # Logging utilities
    │   ├── macros.h          # API export/import macros
    │   ├── speedSam.h        # SpeedSam class definition
    │   └── utils.h           # Utility functions for image handling
    ├── src
    │   ├── engineTRT.cpp     # Implementation of the TensorRT engine
    │   ├── main.cpp          # Main entry point
    │   └── speedSam.cpp      # Implementation of the SpeedSam class
    └── CMakeLists.txt        # CMake configuration
    
# 🚀 Installation
## Prerequisites
    git clone https://github.com/hamdiboukamcha/SPEED-SAM-C-TENSORRT.git
    cd SPEED-SAM-CPP-TENSORRT

    # Create a build directory and compile
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
Note: Update the CMakeLists.txt with the correct paths for TensorRT and OpenCV.

## 📦 Dependencies
    CUDA: NVIDIA's parallel computing platform
    TensorRT: High-performance deep learning inference
    OpenCV: Image processing library
    C++17: Required standard for compilation

# 🔍 Code Overview
## Main Components
    SpeedSam Class (speedSam.h): Manages image encoding and mask decoding.
    EngineTRT Class (engineTRT.h): TensorRT engine creation and inference.
    CUDA Utilities (cuda_utils.h): Macros for CUDA error handling.
    Config (config.h): Defines model parameters and precision settings.
## Key Functions
    EngineTRT::build: Builds the TensorRT engine from an ONNX model.
    EngineTRT::infer: Runs inference on the provided input data.
    SpeedSam::predict: Segments an image using input points or bounding boxes.
## 📞 Contact

For advanced inquiries, feel free to contact me on LinkedIn: <a href="https://www.linkedin.com/in/hamdi-boukamcha/" target="_blank"> <img src="assets/blue-linkedin-logo.png" alt="LinkedIn" width="32" height="32"></a>

## 📜 Citation

If you use this code in your research, please cite the repository as follows:

        @misc{boukamcha2024SpeedSam,
            author = {Hamdi Boukamcha},
            title = {SPEED-SAM-C-TENSORRT},
            year = {2024},
            publisher = {GitHub},
            howpublished = {\url{https://github.com/hamdiboukamcha/SPEED-SAM-C-TENSORRT//}},
        }

    

   

