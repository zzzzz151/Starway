// #include <hip/hip_runtime.h>

__global__ void hidden_to_logits_forward_kernel(
    const float* __restrict__ hiddenLayer,  // [BATCH_SIZE, HIDDEN_SIZE]
    const float* __restrict__ weights,      // [OUTPUT_SIZE, HIDDEN_SIZE]
    const float* __restrict__ biases,       // [OUTPUT_SIZE]
    // legalLogitIdxs elements go from -1 inclusive to OUTPUT_SIZE exclusive
    // legalLogitIdxs is padded with -1
    const int* __restrict__ legalLogitIdxs,  // [BATCH_SIZE, MAX_MOVES_PER_POS]
    float* __restrict__ output               // zeroed, [BATCH_SIZE, MAX_MOVES_PER_POS]
) {
    const int batchSize = gridDim.y;
    const int entryIdx = blockIdx.y;

    const int maxMovesPerPos = gridDim.x;
    const int legalLogitIdxIdx = blockIdx.x;  // 0 inclusive to maxMovesPerPos exclusive

    // 1 thread per hidden neuron
    const int hiddenSize = blockDim.x;
    const int hiddenNeuronIdx = threadIdx.x;

    // clang-format off

    if (entryIdx >= batchSize ||
        legalLogitIdxIdx >= maxMovesPerPos ||
        hiddenNeuronIdx >= hiddenSize) {
        return;
    }

    // clang-format on

    // Indexes 'legalLogitIdxs' and 'output'
    const int idx = entryIdx * maxMovesPerPos + legalLogitIdxIdx;

    // -1 inclusive to OUTPUT_SIZE exclusive
    const int legalLogitIdx = legalLogitIdxs[idx];

    float* outputElem = &output[idx];

    if (legalLogitIdx == -1) {
        if (hiddenNeuronIdx == 0) {
            *outputElem = -10'000.0f;
        }

        return;
    }

    float toAdd = hiddenLayer[entryIdx * hiddenSize + hiddenNeuronIdx] *
                  weights[legalLogitIdx * hiddenSize + hiddenNeuronIdx];

    if (hiddenNeuronIdx == 0) {
        toAdd += biases[legalLogitIdx];
    }

    if (toAdd != 0.0f) {
        atomicAdd(outputElem, toAdd);
    }
}

__global__ void hidden_to_logits_backward_kernel(
    const float* __restrict__ hiddenLayer,  // [BATCH_SIZE, HIDDEN_SIZE]
    const float* __restrict__ weights,      // [OUTPUT_SIZE, HIDDEN_SIZE]
    float* __restrict__ hiddenGrad,         // zeroed, [BATCH_SIZE, HIDDEN_SIZE]
    float* __restrict__ weightsGrad,        // zeroed, [OUTPUT_SIZE, HIDDEN_SIZE]
    float* __restrict__ biasesGrad,         // zeroed, [OUTPUT_SIZE]
    // legalLogitIdxs elements go from -1 inclusive to OUTPUT_SIZE exclusive
    // legalLogitIdxs is padded with -1
    const int* __restrict__ legalLogitIdxs,  // [BATCH_SIZE, MAX_MOVES_PER_POS]
    const float* __restrict__ outputGrad     // [BATCH_SIZE, MAX_MOVES_PER_POS]
) {
    const int batchSize = gridDim.y;
    const int entryIdx = blockIdx.y;

    const int maxMovesPerPos = gridDim.x;
    const int legalLogitIdxIdx = blockIdx.x;  // 0 inclusive to maxMovesPerPos exclusive

    // 1 thread per hidden neuron
    const int hiddenSize = blockDim.x;
    const int hiddenNeuronIdx = threadIdx.x;

    // clang-format off

    if (entryIdx >= batchSize ||
        legalLogitIdxIdx >= maxMovesPerPos ||
        hiddenNeuronIdx >= hiddenSize) {
        return;
    }

    // clang-format on

    // Indexes 'legalLogitIdxs' and 'outputGrad'
    const int idx = entryIdx * maxMovesPerPos + legalLogitIdxIdx;

    // -1 inclusive to OUTPUT_SIZE exclusive
    const int legalLogitIdx = legalLogitIdxs[idx];

    const float myOutputGrad = outputGrad[idx];

    if (legalLogitIdx == -1 || myOutputGrad == 0.0f) {
        return;
    }

    const int hiddenIdx = entryIdx * hiddenSize + hiddenNeuronIdx;
    const int weightsIdx = legalLogitIdx * hiddenSize + hiddenNeuronIdx;

    const float hiddenGradDelta = myOutputGrad * weights[weightsIdx];

    if (hiddenGradDelta != 0.0f) {
        atomicAdd(&hiddenGrad[hiddenIdx], hiddenGradDelta);
    }

    const float weightGradDelta = myOutputGrad * hiddenLayer[hiddenIdx];

    if (weightGradDelta != 0.0f) {
        atomicAdd(&weightsGrad[weightsIdx], weightGradDelta);
    }

    if (hiddenNeuronIdx == 0) {
        atomicAdd(&biasesGrad[legalLogitIdx], myOutputGrad);
    }
}
