#ifndef TRTX_CUDA_UTILS_H_
#define TRTX_CUDA_UTILS_H_

#include <cuda_runtime_api.h>

/// \file trtx_cuda_utils.h
/// \brief Header file providing CUDA utility macros.
///
/// This header file defines utility macros for error checking
/// in CUDA operations, allowing for easier debugging and error
/// handling in GPU-related code.
///
/// \author Hamdi Boukamcha
/// \date 2024

#ifndef CUDA_CHECK
/// \brief Macro for checking CUDA function calls.
///
/// This macro checks the return status of a CUDA call and prints
/// an error message if the call fails. It asserts to halt execution
/// in case of an error.
#define CUDA_CHECK(callstr) \
    { \
        cudaError_t error_code = callstr; \
        if (error_code != cudaSuccess) { \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0); \
        } \
    }
#endif  // CUDA_CHECK

/// \brief Macro for checking conditions with a custom error message.
///
/// This macro checks a condition and, if false, logs an error message
/// and returns a specified value. It includes information about the
/// file, function, and line number where the error occurred.
/// 
/// \param status The condition to check.
/// \param val The value to return if the condition is false.
/// \param errMsg The error message to log.
#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            sample::gLogError << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                    \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

#endif  // TRTX_CUDA_UTILS_H_
