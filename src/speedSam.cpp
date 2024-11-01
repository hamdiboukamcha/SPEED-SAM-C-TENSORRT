#include "speedSam.h"
#include "config.h"

using namespace std;

SpeedSam::SpeedSam(string encoderPath, string decoderPath)
{
    // Initialize the image encoder and mask decoder using the provided model paths
    mImageEncoder = new EngineTRT(encoderPath,
        { "image" },                       // Input names for the encoder
        { "image_embeddings" },           // Output names for the encoder
        false,                             // Not using dynamic shape
        true);                             // Using FP16 precision

    mMaskDecoder = new EngineTRT(decoderPath,
        { "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input" }, // Input names for the decoder
        { "iou_predictions", "low_res_masks" }, // Output names for the decoder
        true,                               // Using dynamic shape
        false);                             // Not using FP16 precision

    // Allocate memory for model features and inputs
    mFeatures = new float[HIDDEN_DIM * FEATURE_WIDTH * FEATURE_HEIGHT];
    mMaskInput = new float[HIDDEN_DIM * HIDDEN_DIM];
    mHasMaskInput = new float;             // Pointer for mask input presence
    mIouPrediction = new float[NUM_LABELS]; // IOU prediction output
    mLowResMasks = new float[NUM_LABELS * HIDDEN_DIM * HIDDEN_DIM]; // Low-resolution masks output
}

SpeedSam::~SpeedSam()
{
    // Clean up dynamically allocated memory
    if (mFeatures)      delete[] mFeatures;
    if (mMaskInput)     delete[] mMaskInput;
    if (mIouPrediction) delete[] mIouPrediction;
    if (mLowResMasks)   delete[] mLowResMasks;

    if (mImageEncoder)  delete mImageEncoder;
    if (mMaskDecoder)   delete mMaskDecoder;
}

Mat SpeedSam::predict(Mat& image, vector<Point> points, vector<float> labels)
{
    // If no points are provided, return an empty mask
    if (points.size() == 0) return cv::Mat(image.rows, image.cols, CV_32FC1);

    // Preprocess the input image for the encoder
    auto resizedImage = resizeImage(image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

    // Perform inference with the image encoder
    mImageEncoder->setInput(resizedImage);
    mImageEncoder->infer();
    mImageEncoder->getOutput(mFeatures);

    // Prepare decoder input data for the specified points
    auto pointData = new float[2 * points.size()]; // Array to hold scaled point coordinates
    prepareDecoderInput(points, pointData, points.size(), image.cols, image.rows);

    // Perform inference with the mask decoder
    mMaskDecoder->setInput(mFeatures, pointData, labels.data(), mMaskInput, mHasMaskInput, points.size());
    mMaskDecoder->infer();
    mMaskDecoder->getOutput(mIouPrediction, mLowResMasks);

    // Post-process the output mask
    Mat imgMask(HIDDEN_DIM, HIDDEN_DIM, CV_32FC1, mLowResMasks);
    upscaleMask(imgMask, image.cols, image.rows); // Upscale to original image size

    delete[] pointData; // Clean up dynamically allocated memory for point data

    return imgMask; // Return the segmented mask
}

void SpeedSam::prepareDecoderInput(vector<Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight)
{
    float scale = MODEL_INPUT_WIDTH / max(imageWidth, imageHeight); // Calculate scaling factor

    // Scale point coordinates
    for (int i = 0; i < numPoints; i++)
    {
        pointData[i * 2] = (float)points[i].x * scale; // X coordinate
        pointData[i * 2 + 1] = (float)points[i].y * scale; // Y coordinate
    }

    // Initialize mask input data
    for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++)
    {
        mMaskInput[i] = 0; // Set mask input to zero
    }
    *mHasMaskInput = 0; // Set has mask input to false
}

Mat SpeedSam::resizeImage(Mat& img, int inputWidth, int inputHeight)
{
    int w, h;
    float aspectRatio = (float)img.cols / (float)img.rows; // Calculate aspect ratio

    // Determine new dimensions while maintaining aspect ratio
    if (aspectRatio >= 1)
    {
        w = inputWidth;
        h = int(inputHeight / aspectRatio);
    }
    else
    {
        w = int(inputWidth * aspectRatio);
        h = inputHeight;
    }

    // Create a new image with the new size
    Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, INTER_LINEAR); // Resize the original image
    Mat out(inputHeight, inputWidth, CV_8UC3, 0.0); // Initialize output image
    re.copyTo(out(Rect(0, 0, re.cols, re.rows))); // Copy resized image to output

    return out; // Return the resized image
}

void SpeedSam::upscaleMask(Mat& mask, int targetWidth, int targetHeight, int size)
{
    int limX, limY;
    // Calculate limits for upscaling based on target dimensions
    if (targetWidth > targetHeight)
    {
        limX = size;
        limY = size * targetHeight / targetWidth;
    }
    else
    {
        limX = size * targetWidth / targetHeight;
        limY = size;
    }

    // Resize the mask to the target dimensions
    cv::resize(mask(Rect(0, 0, limX, limY)), mask, Size(targetWidth, targetHeight));
}
