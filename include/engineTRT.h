#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

/// \class TRTModule
/// \brief A class for handling TensorRT model inference.
///
/// This class manages loading, setting inputs, and executing inference 
/// for a TensorRT model. It provides methods for setting input data,
/// retrieving output predictions, and handling both static and dynamic shapes.
///
/// \author Hamdi Boukamcha
/// \date 2024
class EngineTRT
{

public:
    /// \brief Constructor for the TRTModule class.
    /// 
    /// \param modelPath Path to the ONNX model file.
    /// \param inputNames Names of the input tensors.
    /// \param outputNames Names of the output tensors.
    /// \param isDynamicShape Indicates if the model uses dynamic shapes.
    /// \param isFP16 Indicates if the model should use FP16 precision.
    EngineTRT(string modelPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16);

    /// \brief Performs inference on the input data.
    /// \return True if inference was successful, false otherwise.
    bool infer();

    /// \brief Sets the input image for inference.
    /// 
    /// \param image The input image to be processed.
    void setInput(Mat& image);

    /// \brief Sets multiple inputs for inference from raw data.
    /// 
    /// \param features Pointer to feature data.
    /// \param imagePointCoords Pointer to image point coordinates.
    /// \param imagePointLabels Pointer to image point labels.
    /// \param maskInput Pointer to mask input data.
    /// \param hasMaskInput Pointer to existence of mask data.
    /// \param numPoints The number of points in the input.
    void setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints);

    /// \brief Retrieves the output predictions for IoU and low-resolution masks.
    /// 
    /// \param iouPrediction Pointer to store IoU prediction output.
    /// \param lowResolutionMasks Pointer to store low-resolution mask output.
    void getOutput(float* iouPrediction, float* lowResolutionMasks);

    /// \brief Retrieves the output features from inference.
    /// 
    /// \param features Pointer to store output features.
    void getOutput(float* features);

    /// \brief Destructor for the TRTModule class.
    ~EngineTRT();

private:
    /// \brief Builds the TensorRT engine from an ONNX model.
    /// 
    /// \param onnxPath Path to the ONNX model file.
    /// \param inputNames Names of the input tensors.
    /// \param outputNames Names of the output tensors.
    /// \param isDynamicShape Indicates if the model uses dynamic shapes.
    /// \param isFP16 Indicates if the model should use FP16 precision.
    void build(string onnxPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape = false, bool isFP16 = false);

    void saveEngine(const std::string& engineFilePath);

    /// \brief Deserializes the engine from a file.
    /// 
    /// \param engineName Name of the engine file.
    /// \param inputNames Names of the input tensors.
    /// \param outputNames Names of the output tensors.
    void deserializeEngine(string engineName, vector<string> inputNames, vector<string> outputNames);

    /// \brief Initializes the TensorRT module with input and output names.
    /// 
    /// \param inputNames Names of the input tensors.
    /// \param outputNames Names of the output tensors.
    void initialize(vector<string> inputNames, vector<string> outputNames);

    /// \brief Gets the size of a buffer based on its dimensions.
    /// 
    /// \param dims The dimensions of the buffer.
    /// \return The size in bytes of the buffer.
    size_t getSizeByDim(const Dims& dims);

    /// \brief Copies buffers between device and host memory.
    /// 
    /// \param copyInput Indicates whether to copy input data.
    /// \param deviceToHost Indicates the direction of the copy.
    /// \param async Indicates whether the copy should be asynchronous.
    /// \param stream The CUDA stream to use for the copy operation.
    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0);

    /// \brief Asynchronously copies input data to the device.
    /// 
    /// \param stream The CUDA stream to use for the copy operation.
    void copyInputToDeviceAsync(const cudaStream_t& stream = 0);

    /// \brief Asynchronously copies output data from the device to host.
    /// 
    /// \param stream The CUDA stream to use for the copy operation.
    void copyOutputToHostAsync(const cudaStream_t& stream = 0);

    vector<Dims> mInputDims;            //!< The dimensions of the input to the network.
    vector<Dims> mOutputDims;           //!< The dimensions of the output to the network.
    vector<void*> mGpuBuffers;          //!< The vector of device buffers needed for engine execution.
    vector<float*> mCpuBuffers;         //!< The vector of CPU buffers for input/output.
    vector<size_t> mBufferBindingBytes; //!< The sizes in bytes of each buffer binding.
    vector<size_t> mBufferBindingSizes; //!< The sizes of the buffer bindings.
    cudaStream_t mCudaStream;           //!< The CUDA stream used for asynchronous operations.

    IRuntime* mRuntime;                 //!< The TensorRT runtime used to deserialize the engine.
    ICudaEngine* mEngine;               //!< The TensorRT engine used to run the network.
    IExecutionContext* mContext;        //!< The context for executing inference using an ICudaEngine.
};
