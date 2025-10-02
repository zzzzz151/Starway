#include <future>
#include <iostream>
#include <print>
#include <vector>

#include "../converter/data_entry.hpp"
#include "../utils.hpp"
#include "batch.hpp"
#include "worker.hpp"

// Needed to export functions on Windows
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API
#endif

std::vector<Worker> gWorkers = {};
i32 gWorkerIdx = -1;

extern "C" API void init(const char* dataFilePath,
                         const size_t batchSize,
                         const size_t numThreads) {
    assert(batchSize > 0);
    assert(numThreads > 0);

    // Open data file
    std::ifstream dataFile(dataFilePath, std::ios::binary | std::ios::ate);
    assert(dataFile);

    // Assert file has at least 1 batch for each thread
    const i64 fileSizeBytes = dataFile.tellg();
    assert(fileSizeBytes >= static_cast<i64>(numThreads * batchSize * sizeof(StarwayDataEntry)));

    // Assert file doesn't end in the middle of a data entry
    assert(static_cast<size_t>(fileSizeBytes) % sizeof(StarwayDataEntry) == 0);

    // Assert file ends with a full batch of data entries
    assert((static_cast<size_t>(fileSizeBytes) / sizeof(StarwayDataEntry)) % batchSize == 0);

    // Allocate workers
    for (size_t i = 0; i < numThreads; i++) {
        gWorkers.push_back(Worker(i, dataFilePath, static_cast<size_t>(fileSizeBytes), batchSize));
    }

    // Make workers start working
    for (Worker& worker : gWorkers) {
        worker.mFuture = std::async(
            std::launch::async, &Worker::getNextBatch, &worker, gWorkers.size(), batchSize);
    }
}

extern "C" API Batch* next_batch(const size_t batchSize) {
    assert(gWorkers.size() > 0);

    // After the first next_batch() call, make the last used worker load his next batch async
    // since the last batch is no longer being used by PyTorch
    if (gWorkerIdx != -1) {
        Worker& worker = gWorkers[static_cast<size_t>(gWorkerIdx)];

        worker.mFuture = std::async(
            std::launch::async, &Worker::getNextBatch, &worker, gWorkers.size(), batchSize);
    }

    gWorkerIdx = (gWorkerIdx + 1) % static_cast<i32>(gWorkers.size());

    Worker& worker = gWorkers[static_cast<size_t>(gWorkerIdx)];

    return worker.mFuture.get();
}

int main() {
    std::println("Dataloader main()");
    return 0;
}
