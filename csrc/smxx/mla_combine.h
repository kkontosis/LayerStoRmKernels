#pragma once

#include "params.h"

template<typename ElementT>
void run_mla_combine_kernel(MlaCombineParams &params, cudaStream_t stream);
