/*
 * cuda_version_control.h
 *
 *  Created on: Nov 3, 2013
 *  Author: Ting-Shuo Chou <tingshuc@uci.edu>
 */

#ifndef _CUDA_VERSION_CONTROL_H_
#define _CUDA_VERSION_CONTROL_H_

#if __CUDA3__
    #include <cutil_inline.h>
    #define CUDA_CHECK_ERRORS(x) cutilSafeCall(x)
    #define CUDA_CHECK_ERRORS_MACRO(x) CUDA_SAFE_CALL(x)

    #define CUDA_GET_LAST_ERROR(x) cutilCheckMsg(x)
    #define CUDA_GET_LAST_ERROR_MACRO(x) CUT_CHECK_ERROR(x)

    #define CUDA_CREATE_TIMER(x) cutCreateTimer(&(x))
    #define CUDA_DELETE_TIMER(x) cutDeleteTimer(x)
    #define CUDA_RESET_TIMER(x) cutResetTimer(x)
    #define CUDA_START_TIMER(x) cutStartTimer(x)
    #define CUDA_STOP_TIMER(x) cutStopTimer(x)
    #define CUDA_GET_TIMER_VALUE(x) cutGetTimerValue(x)

    #define CUDA_GET_MAXGFLOP_DEVICE_ID cutGetMaxGflopsDeviceId
#elif defined(__CUDA5__) || defined(__CUDA6__)
    #include <helper_cuda.h>
    #define CUDA_CHECK_ERRORS(x) checkCudaErrors(x)
    #define CUDA_CHECK_ERRORS_MACRO(x) checkCudaErrors(x)

    #define CUDA_CREATE_TIMER(x) sdkCreateTimer(&(x))
    #define CUDA_DELETE_TIMER(x) sdkDeleteTimer(&(x))
    #define CUDA_RESET_TIMER(x) sdkResetTimer(&(x))
    #define CUDA_START_TIMER(x) sdkStartTimer(&(x))
    #define CUDA_STOP_TIMER(x) sdkStopTimer(&(x))
    #define CUDA_GET_TIMER_VALUE(x) sdkGetTimerValue(&(x))

    #define CUDA_GET_LAST_ERROR(x) getLastCudaError(x)
    #define CUDA_GET_LAST_ERROR_MACRO(x) getLastCudaError(x)

    #define CUDA_GET_MAXGFLOP_DEVICE_ID gpuGetMaxGflopsDeviceId
#endif


#endif /* _CUDA_VERSION_CONTROL_H_ */