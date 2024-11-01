#pragma once

/// \file model_params.h
/// \brief Header file defining model parameters and configuration macros.
///
/// This header file contains macros for model parameters including input dimensions,
/// feature dimensions, and precision settings.
///
/// \author Hamdi Boukamcha
/// \date 2024

#define USE_FP16  ///< Set to use FP16 (float16) precision, or comment to use FP32 (float32) precision.

#define MAX_NUM_PROMPTS  1  ///< Maximum number of prompts to be processed at once.

// Model Params
#define MODEL_INPUT_WIDTH  1024.0f  ///< Width of the model input in pixels.
#define MODEL_INPUT_HEIGHT 1024.0f  ///< Height of the model input in pixels.
#define HIDDEN_DIM         256       ///< Dimension of the hidden layer.
#define NUM_LABELS        4         ///< Number of output labels.
#define FEATURE_WIDTH      64        ///< Width of the feature map.
#define FEATURE_HEIGHT     64        ///< Height of the feature map.
