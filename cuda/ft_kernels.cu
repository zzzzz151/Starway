// #include <hip/hip_runtime.h>

constexpr int MAX_ACTIVE_FEATURES = 32;

__global__ void ft_forward_kernel(
    const int* __restrict__ activeFeatures,  // [BATCH_SIZE, MAX_ACTIVE_FEATURES] (padded with -1)
    const float* __restrict__ weights,       // [INPUT_SIZE, HIDDEN_SIZE]
    const float* __restrict__ biases,        // [HIDDEN_SIZE]
    float* __restrict__ output)              // uninitialized, [BATCH_SIZE, HIDDEN_SIZE]
{
    // 1 block per data entry in the batch
    const int batchSize = gridDim.x;
    const int entryIdx = blockIdx.x;

    // 1 thread per hidden neuron
    const int hiddenSize = blockDim.x;
    const int hiddenNeuronIdx = threadIdx.x;

    if (entryIdx >= batchSize || hiddenNeuronIdx >= hiddenSize) {
        return;
    }

    float* hiddenNeuron = &output[entryIdx * hiddenSize + hiddenNeuronIdx];

    *hiddenNeuron = biases[hiddenNeuronIdx];

    for (int i = 0; i < MAX_ACTIVE_FEATURES; i++) {
        const int featureIdx = activeFeatures[entryIdx * MAX_ACTIVE_FEATURES + i];

        // Null terminator (no more active pieces in this data entry)
        if (featureIdx == -1) {
            break;
        }

        *hiddenNeuron += weights[featureIdx * hiddenSize + hiddenNeuronIdx];
    }
}

__global__ void ft_backward_kernel(
    const int* __restrict__ activeFeatures,  // [BATCH_SIZE, MAX_ACTIVE_FEATURES] (padded with -1)
    float* __restrict__ weightsGrad,         // zeroed, [INPUT_SIZE, HIDDEN_SIZE]
    float* __restrict__ biasesGrad,          // zeroed, [HIDDEN_SIZE]
    const float* __restrict__ outputGrad)    // [BATCH_SIZE, HIDDEN_SIZE]
{
    // 1 block per data entry in the batch
    const int batchSize = gridDim.x;
    const int entryIdx = blockIdx.x;

    // 1 thread per hidden neuron
    const int hiddenSize = blockDim.x;
    const int hiddenNeuronIdx = threadIdx.x;

    if (entryIdx >= batchSize || hiddenNeuronIdx >= hiddenSize) {
        return;
    }

    const float myOutputGrad = outputGrad[entryIdx * hiddenSize + hiddenNeuronIdx];

    atomicAdd(&biasesGrad[hiddenNeuronIdx], myOutputGrad);

    for (int i = 0; i < MAX_ACTIVE_FEATURES; i++) {
        const int featureIdx = activeFeatures[entryIdx * MAX_ACTIVE_FEATURES + i];

        // Null terminator (no more active pieces in this data entry)
        if (featureIdx == -1) {
            break;
        }

        atomicAdd(&weightsGrad[featureIdx * hiddenSize + hiddenNeuronIdx], myOutputGrad);
    }
}
