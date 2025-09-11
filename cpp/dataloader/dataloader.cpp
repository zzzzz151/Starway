#include <iostream>

#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../converter/data_entry.hpp"
#include "../utils.hpp"
#include "batch.hpp"
#include "move_mapping.hpp"

// Needed to export functions on Windows
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

// These constants are set in init(), which is called from train.py
std::string DATA_FILE_PATH = "";
size_t DATA_FILE_BYTES = 0;
size_t BATCH_SIZE = 0;
size_t NUM_THREADS = 0;

std::vector<Batch> gBatches = {};  // NUM_THREADS batches
size_t gNextBatchIdx = 0;          // 0 to NUM_THREADS-1, including both
size_t gDataFilePos = 0;

extern "C" API void init(const char* dataFilePath, const i32 batchSize, const i32 numThreads) {
    std::cout << "Batches in data file: " << BATCH_POSITIONS.size() << std::endl;

    DATA_FILE_PATH = static_cast<std::string>(dataFilePath);

    // Open file in binary mode and at the end
    std::ifstream dataFile(DATA_FILE_PATH, std::ios::binary | std::ios::ate);

    if (!dataFile) {
        std::cerr << "Error opening file " << DATA_FILE_PATH << std::endl;
        exit(EXIT_FAILURE);
    }

    if (batchSize <= 0) {
        std::cerr << "Batch size must be > 0 but is " << batchSize << std::endl;
        exit(EXIT_FAILURE);
    }

    if (numThreads <= 0) {
        std::cerr << "Threads count must be > 0 but is " << numThreads << std::endl;
        exit(EXIT_FAILURE);
    }

    DATA_FILE_BYTES = static_cast<size_t>(dataFile.tellg());
    BATCH_SIZE = static_cast<size_t>(batchSize);
    NUM_THREADS = static_cast<size_t>(numThreads);

    for (size_t i = 0; i < NUM_THREADS; i++) {
        gBatches.push_back(Batch(BATCH_SIZE));
    }
}

extern "C" API Batch* nextBatch() { return nullptr; }

int main() {
    std::cout << "Dataloader main()" << std::endl;
    return 0;
}
