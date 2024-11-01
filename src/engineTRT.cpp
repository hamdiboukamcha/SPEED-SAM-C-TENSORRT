#include "engineTRT.h"
#include "logging.h"
#include "cuda_utils.h"
#include "config.h"
#include "macros.h"
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

static Logger gLogger;

std::string getFileExtension(const std::string& filePath) {
    // Find the position of the last dot in the file path
    size_t dotPos = filePath.find_last_of(".");
    // If a dot is found, extract and return the substring after the dot as the file extension
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos + 1);
    }
    // Return an empty string if no extension is found
    return "";
}

EngineTRT::EngineTRT(string modelPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16) {
    // Check if the model file has an ".onnx" extension
    if (getFileExtension(modelPath) == "onnx") {
        // If the file is an ONNX model, build the engine using the provided parameters
        cout << "Building Engine from " << modelPath << endl;
        build(modelPath, inputNames, outputNames, isDynamicShape, isFP16);
    }
    else {
        // If the file is not an ONNX model, deserialize an existing engine
        cout << "Deserializing Engine." << endl;
        deserializeEngine(modelPath, inputNames, outputNames);
    }
}

EngineTRT::~EngineTRT() {
    // Release the CUDA stream
    cudaStreamDestroy(mCudaStream);
    // Free GPU buffers allocated for inference
    for (int i = 0; i < mGpuBuffers.size(); i++)
        CUDA_CHECK(cudaFree(mGpuBuffers[i]));
    // Free CPU buffers
    for (int i = 0; i < mCpuBuffers.size(); i++)
        delete[] mCpuBuffers[i];

    // Clean up and destroy the TensorRT engine components
    delete mContext;  // Destroy the execution context
    delete mEngine;   // Destroy the engine
    delete mRuntime;  // Destroy the runtime
}

void EngineTRT::build(string onnxPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16)
{
    // Check if the ONNX file exists. If not, print an error message and return.
    if (!std::filesystem::exists(onnxPath)) {
        std::cerr << "ONNX file not found: " << onnxPath << std::endl;
        return;  // Early exit if the ONNX file is missing
    }

    // Create an inference builder for building the TensorRT engine.
    auto builder = createInferBuilder(gLogger);
    assert(builder != nullptr);  // Ensure the builder is created successfully

    // Use explicit batch size, which is needed for ONNX models.
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);  // Ensure the network is created successfully

    // Create a builder configuration object to set options like FP16 precision.
    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);  // Ensure the config is created successfully

    // If dynamic shape support is needed, configure the optimization profile.
    if (isDynamicShape) // Only designed for NanoSAM mask decoder
    {
        // Create an optimization profile for dynamic input shapes.
        auto profile = builder->createOptimizationProfile();

        // Set the minimum, optimal, and maximum dimensions for the first input.
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMIN, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kOPT, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMAX, Dims3{ 1, 10, 2 });

        // Set the minimum, optimal, and maximum dimensions for the second input.
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMIN, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kOPT, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMAX, Dims2{ 1, 10 });

        // Add the optimization profile to the builder configuration.
        config->addOptimizationProfile(profile);
    }

    // Enable FP16 mode if specified.
    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);  // Use mixed precision for faster inference
    }

    // Create a parser to convert the ONNX model to a TensorRT network.
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    assert(parser != nullptr);  // Ensure the parser is created successfully

    // Parse the ONNX model from the specified file.
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));

    // Ensure the CUDA stream used for profiling is valid.
    assert(mCudaStream != nullptr);

    // Serialize the built network into a binary plan for execution.
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    assert(plan != nullptr);  // Ensure the network was serialized successfully

    // Create a runtime object for deserializing the engine.
    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime != nullptr);  // Ensure the runtime is created successfully

    // Deserialize the serialized plan to create an execution engine.
    mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size(), nullptr);
    assert(mEngine != nullptr);  // Ensure the engine was deserialized successfully

    // Create an execution context for running inference.
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);  // Ensure the context is created successfully

    // Clean up resources.
    delete network;
    delete config;
    delete parser;
    delete plan;

    // Initialize the engine with the input and output names.
    initialize(inputNames, outputNames);
}

void EngineTRT::saveEngine(const std::string& engineFilePath) {
    if (mEngine) {
        // Serialize the engine to a binary format.
        IHostMemory* serializedEngine = mEngine->serialize();
        std::ofstream engineFile(engineFilePath, std::ios::binary);
        if (engineFile) {
            // Write the serialized engine data to the specified file.
            engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
            std::cout << "Serialized engine saved to " << engineFilePath << std::endl;
        }
        serializedEngine->destroy();  // Destroy the serialized engine memory
    }
}

void EngineTRT::deserializeEngine(string engine_name, vector<string> inputNames, vector<string> outputNames)
{
    // Open the engine file in binary mode.
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);  // Trigger an assertion failure if the file cannot be opened
    }

    // Determine the size of the file and read the serialized engine data.
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    assert(serializedEngine);  // Ensure memory allocation was successful
    file.read(serializedEngine, size);
    file.close();

    // Create a runtime object and deserialize the engine.
    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime);  // Ensure the runtime is created successfully
    mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);
    mContext = mEngine->createExecutionContext();
    delete[] serializedEngine;  // Free the serialized engine memory

    // Ensure the number of bindings matches the expected number of inputs and outputs.
    assert(mEngine->getNbBindings() == inputNames.size() + outputNames.size());

    // Initialize the engine with the input and output names.
    initialize(inputNames, outputNames);
}

void EngineTRT::initialize(vector<string> inputNames, vector<string> outputNames)
{
    // Loop through the input names and get the corresponding binding index from the TensorRT engine
    for (int i = 0; i < inputNames.size(); i++)
    {
        const int inputIndex = mEngine->getBindingIndex(inputNames[i].c_str());
    }

    // Loop through the output names and get the corresponding binding index from the TensorRT engine
    for (int i = 0; i < outputNames.size(); i++)
    {
        const int outputIndex = mEngine->getBindingIndex(outputNames[i].c_str());
    }

    // Resize the GPU and CPU buffer vectors to accommodate all the engine bindings
    mGpuBuffers.resize(mEngine->getNbBindings());
    mCpuBuffers.resize(mEngine->getNbBindings());

    // Loop through all bindings to allocate memory and store dimension information
    for (size_t i = 0; i < mEngine->getNbBindings(); ++i)
    {
        // Calculate the size required for the binding based on its dimensions
        size_t binding_size = getSizeByDim(mEngine->getBindingDimensions(i));
        mBufferBindingSizes.push_back(binding_size);  // Store the size of the binding
        mBufferBindingBytes.push_back(binding_size * sizeof(float));  // Calculate the size in bytes

        // Allocate host memory for the CPU buffer
        mCpuBuffers[i] = new float[binding_size];

        // Allocate device memory for the GPU buffer
        cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]);

        // Store input and output dimensions separately based on whether the binding is an input or output
        if (mEngine->bindingIsInput(i))
        {
            mInputDims.push_back(mEngine->getBindingDimensions(i));
        }
        else
        {
            mOutputDims.push_back(mEngine->getBindingDimensions(i));
        }
    }

    // Create a CUDA stream for asynchronous operations
    CUDA_CHECK(cudaStreamCreate(&mCudaStream));
}

bool EngineTRT::infer()
{
    // Copy data from host (CPU) input buffers to device (GPU) input buffers asynchronously
    copyInputToDeviceAsync(mCudaStream);

    // Perform inference using TensorRT, passing the GPU buffers
    bool status = mContext->executeV2(mGpuBuffers.data());

    if (!status)
    {
        // If inference fails, print an error message and return false
        cout << "inference error!" << endl;
        return false;
    }

    // Copy the results from device (GPU) output buffers to host (CPU) output buffers asynchronously
    copyOutputToHostAsync(mCudaStream);

    // Return true if inference was successful
    return true;
}

void EngineTRT::copyInputToDeviceAsync(const cudaStream_t& stream)
{
    // Perform asynchronous memory copy from CPU to GPU for the input buffers
    memcpyBuffers(true, false, true, stream);
}

void EngineTRT::copyOutputToHostAsync(const cudaStream_t& stream)
{
    // Calls memcpyBuffers to handle the copying of data from GPU to CPU memory.
    // Arguments: false (do not copy input buffers), true (copy data from device to host),
    // true (perform the copy asynchronously), and the given CUDA stream.
    memcpyBuffers(false, true, true, stream);
}

void EngineTRT::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
    // Loop through all bindings (inputs and outputs) in the TensorRT engine.
    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        // Determine the destination and source pointers based on the copy direction.
        void* dstPtr = deviceToHost ? mCpuBuffers[i] : mGpuBuffers[i];
        const void* srcPtr = deviceToHost ? mGpuBuffers[i] : mCpuBuffers[i];
        // Get the size of the buffer in bytes.
        const size_t byteSize = mBufferBindingBytes[i];
        // Set the type of memory copy operation based on the direction.
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        // Check if the current binding is an input or output and copy accordingly.
        if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
        {
            if (async)
            {
                // Perform asynchronous memory copy using the CUDA stream.
                CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
            }
            else
            {
                // Perform synchronous memory copy.
                CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }
}

size_t EngineTRT::getSizeByDim(const Dims& dims)
{
    size_t size = 1;

    // Loop through each dimension and multiply to calculate the total size.
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        // If the dimension is -1 (dynamic), use a predefined maximum size.
        if (dims.d[i] == -1)
            size *= MAX_NUM_PROMPTS;
        else
            size *= dims.d[i];
    }

    return size;
}

void EngineTRT::setInput(Mat& image)
{
    // Extract input dimensions (height and width) from the model's input shape
    const int inputH = mInputDims[0].d[2];
    const int inputW = mInputDims[0].d[3];

    int i = 0;  // Index counter for buffer placement

    // Iterate over each pixel in the input image
    for (int row = 0; row < image.rows; ++row)
    {
        // Pointer to the start of the row in the image data
        uchar* uc_pixel = image.data + row * image.step;

        for (int col = 0; col < image.cols; ++col)
        {
            // Normalizing the pixel values for the RGB channels
            // Convert the BGR image to normalized RGB and store in mCpuBuffers
            mCpuBuffers[0][i] = ((float)uc_pixel[2] / 255.0f - 0.485f) / 0.229f; // Red channel
            mCpuBuffers[0][i + image.rows * image.cols] = ((float)uc_pixel[1] / 255.0f - 0.456f) / 0.224f; // Green channel
            mCpuBuffers[0][i + 2 * image.rows * image.cols] = ((float)uc_pixel[0] / 255.0f - 0.406f) / 0.225f; // Blue channel

            uc_pixel += 3;  // Move to the next pixel
            ++i;  // Increment index
        }
    }
}

void EngineTRT::setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints)
{
    // Clean up old buffers and allocate new buffers for the input data
    delete[] mCpuBuffers[1];
    delete[] mCpuBuffers[2];
    mCpuBuffers[1] = new float[numPoints * 2];  // Buffer for point coordinates
    mCpuBuffers[2] = new float[numPoints];      // Buffer for point labels

    // Allocate memory on the GPU for the input data
    cudaMalloc(&mGpuBuffers[1], sizeof(float) * numPoints * 2); // Coordinates
    cudaMalloc(&mGpuBuffers[2], sizeof(float) * numPoints);     // Labels

    // Set the size of the data binding in bytes for TensorRT
    mBufferBindingBytes[1] = sizeof(float) * numPoints * 2;
    mBufferBindingBytes[2] = sizeof(float) * numPoints;

    // Copy input data into CPU buffers
    memcpy(mCpuBuffers[0], features, mBufferBindingBytes[0]);
    memcpy(mCpuBuffers[1], imagePointCoords, sizeof(float) * numPoints * 2);
    memcpy(mCpuBuffers[2], imagePointLabels, sizeof(float) * numPoints);
    memcpy(mCpuBuffers[3], maskInput, mBufferBindingBytes[3]);
    memcpy(mCpuBuffers[4], hasMaskInput, mBufferBindingBytes[4]);

    // Configure TensorRT to use a dynamic input shape
    mContext->setOptimizationProfileAsync(0, mCudaStream); // Set the optimization profile
    mContext->setBindingDimensions(1, Dims3{ 1, numPoints, 2 }); // Set input dimensions for coordinates
    mContext->setBindingDimensions(2, Dims2{ 1, numPoints });    // Set input dimensions for labels
}

void EngineTRT::getOutput(float* features)
{
    // Copy the output features from the CPU buffer to the provided memory
    memcpy(features, mCpuBuffers[1], mBufferBindingBytes[1]);
}

void EngineTRT::getOutput(float* iouPrediction, float* lowResolutionMasks)
{
    // Copy the low-resolution masks and IOU predictions from the CPU buffers
    memcpy(lowResolutionMasks, mCpuBuffers[5], mBufferBindingBytes[5]);
    memcpy(iouPrediction, mCpuBuffers[6], mBufferBindingBytes[6]);
}

