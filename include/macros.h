#ifndef __MACROS_H
#define __MACROS_H

/// \file macros.h
/// \brief Header file defining macros for API export and compatibility.
///
/// This header file contains macros that facilitate 
/// the export and import of functions for shared libraries,
/// as well as compatibility settings based on the NV TensorRT version.
///
/// \author Hamdi Boukamcha
/// \date 2024

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport) ///< Macro for exporting functions in Windows.
#else
#define API __attribute__((visibility("default"))) ///< Macro for exporting functions in non-Windows environments.
#endif
#else
#if defined(_MSC_VER)
#define API __declspec(dllimport) ///< Macro for importing functions in Windows.
#else
#define API ///< No import/export in non-Windows environments.
#endif
#endif  // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept ///< Macro for noexcept specification based on TensorRT version.
#define TRT_CONST_ENQUEUE const ///< Macro to define const enqueue for TensorRT version >= 8.
#else
#define TRT_NOEXCEPT ///< No exception specification for TensorRT version < 8.
#define TRT_CONST_ENQUEUE ///< No const enqueue definition for TensorRT version < 8.
#endif

#endif  // __MACROS_H
