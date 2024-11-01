#pragma once

#include <string>
#include "engineTRT.h"

/// \class NanoSam
/// \brief A class for handling image predictions with an encoder and decoder model.
///
/// This class manages the process of encoding and decoding images, 
/// allowing for the prediction of masks based on input images and points.
///
/// \author Hamdi Boukamcha
/// \date 2024
class SpeedSam
{

public:
    /// \brief Constructor for the NanoSam class.
    /// 
    /// \param encoderPath Path to the encoder model.
    /// \param decoderPath Path to the decoder model.
    SpeedSam(std::string encoderPath, std::string decoderPath);

    /// \brief Destructor for the NanoSam class.
    ~SpeedSam();

    /// \brief Predicts masks based on the input image and points.
    /// 
    /// \param image The input image for prediction.
    /// \param points The points used for mask prediction.
    /// \param labels The labels associated with the points.
    /// \return A matrix containing the predicted masks.
    Mat predict(Mat& image, std::vector<Point> points, std::vector<float> labels);

private:
    // Variables
    float* mFeatures;        ///< Pointer to the feature data.
    float* mMaskInput;      ///< Pointer to the mask input data.
    float* mHasMaskInput;   ///< Pointer to the mask existence input data.
    float* mIouPrediction;   ///< Pointer to the IoU prediction data.
    float* mLowResMasks;     ///< Pointer to the low-resolution masks.

    EngineTRT* mImageEncoder;  ///< Pointer to the image encoder module.
    EngineTRT* mMaskDecoder;    ///< Pointer to the mask decoder module.

    /// \brief Upscales the given mask to the target width and height.
    /// 
    /// \param mask The mask to upscale.
    /// \param targetWidth The target width for upscaling.
    /// \param targetHeight The target height for upscaling.
    /// \param size The size of the mask (default is 256).
    void upscaleMask(Mat& mask, int targetWidth, int targetHeight, int size = 256);

    /// \brief Resizes the input image to match the model's dimensions.
    /// 
    /// \param img The image to resize.
    /// \param modelWidth The width required by the model.
    /// \param modelHeight The height required by the model.
    /// \return A resized image matrix.
    Mat resizeImage(Mat& img, int modelWidth, int modelHeight);

    /// \brief Prepares the decoder input from the provided points.
    /// 
    /// \param points The points to be converted into input data.
    /// \param pointData Pointer to the data array for points.
    /// \param numPoints The number of points.
    /// \param imageWidth The width of the input image.
    /// \param imageHeight The height of the input image.
    void prepareDecoderInput(std::vector<Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight);
};
