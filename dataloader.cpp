// clang-format off

#include "dataloader.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <thread>

// These constants are set in init(), which is called from train.py
std::string DATA_FILE_PATH = "";
size_t DATA_FILE_BYTES = 0;
size_t BATCH_SIZE = 0;
size_t NUM_THREADS = 0;

std::vector<Batch> gBatches = { }; // NUM_THREADS batches
size_t gNextBatchIdx = 0; // 0 to NUM_THREADS-1
size_t gDataFilePos = 0;

extern "C" API void init(
    const char* dataFilePath, const i32 batchSize, const i32 numThreads)
{
    DATA_FILE_PATH = static_cast<std::string>(dataFilePath);

    // Open file in binary mode and at the end
    std::ifstream dataFile(DATA_FILE_PATH, std::ios::binary | std::ios::ate);

    if (!dataFile || !dataFile.is_open())
    {
        std::cout << "Error opening file " << DATA_FILE_PATH << std::endl;
        exit(EXIT_FAILURE);
    }

    if (batchSize <= 0)
    {
        std::cout << "Batch size must be > 0 but is " << batchSize << std::endl;
        exit(EXIT_FAILURE);
    }

    if (numThreads <= 0)
    {
        std::cout << "Threads count must be > 0 but is " << numThreads << std::endl;
        exit(EXIT_FAILURE);
    }

    BATCH_SIZE = static_cast<size_t>(batchSize);
    NUM_THREADS = static_cast<size_t>(numThreads);

    DATA_FILE_BYTES = static_cast<size_t>(dataFile.tellg());

    if (DATA_FILE_BYTES % static_cast<size_t>(sizeof(DataEntry)) != 0)
    {
        std::cout << "Data file bytes must divide data entry bytes but it doesn't" << std::endl;
        exit(EXIT_FAILURE);
    }

    const size_t numDataEntries = DATA_FILE_BYTES / static_cast<size_t>(sizeof(DataEntry));

    if (numDataEntries % BATCH_SIZE != 0)
    {
        std::cout << "Data entries count must divide batch size but it doesn't" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NUM_THREADS; i++)
        gBatches.push_back(Batch(BATCH_SIZE));
}

inline void loadBatch(const size_t threadId)
{
    // Open file at correct position

    size_t dataFilePos
        = gDataFilePos + static_cast<size_t>(sizeof(DataEntry)) * BATCH_SIZE * threadId;

    if (dataFilePos >= DATA_FILE_BYTES)
        dataFilePos -= DATA_FILE_BYTES;

    // Open data file in correct position
    std::ifstream dataFile(DATA_FILE_PATH, std::ios::binary);
    assert(dataFile && dataFile.is_open());
    dataFile.seekg(static_cast<i64>(dataFilePos), std::ios::beg);

    // Fill the batch gBatches[threadId]

    Batch& batch = gBatches[threadId];
    batch.numActiveFeatures = 0;

    DataEntry dataEntry;

    const auto featureIdx = [] (
        const auto pieceColor,
        const auto pieceType,
        const auto square,
        const auto kingSquare) constexpr -> size_t
    {
        const bool flipVAxis = static_cast<size_t>(kingSquare) % 8 > 3;

        return static_cast<size_t>(pieceColor) * 384
             + static_cast<size_t>(pieceType) * 64
             + static_cast<size_t>(square) ^ (flipVAxis ? 7 : 0);
    };

    for (size_t entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++)
    {
        dataFile.read(reinterpret_cast<char*>(&dataEntry), sizeof(DataEntry));

        batch.isWhiteStm[entryIdx] = dataEntry.isWhiteStm;

        while (dataEntry.occupied > 0)
        {
            const auto square = popLsb(dataEntry.occupied);
            const auto pieceColor = dataEntry.pieces & 0b1;
            const auto pieceType = (dataEntry.pieces & 0b1110) >> 1;

            const size_t idx = batch.numActiveFeatures * 2;

            batch.activeFeaturesWhite[idx] = static_cast<i16>(entryIdx);
            batch.activeFeaturesBlack[idx] = static_cast<i16>(entryIdx);

            batch.activeFeaturesWhite[idx + 1] = static_cast<i16>(
                featureIdx(pieceColor, pieceType, square, dataEntry.whiteKingSquare)
            );

            batch.activeFeaturesBlack[idx + 1] = static_cast<i16>(
                featureIdx(pieceColor, pieceType, square, dataEntry.blackKingSquare)
            );

            batch.numActiveFeatures++;
            dataEntry.pieces >>= 4;
        }

        batch.stmScores[entryIdx] = dataEntry.stmScore;
        batch.stmWDLs[entryIdx] = static_cast<float>(dataEntry.stmWdl + 1) / 2.0f;
    }
}

extern "C" API Batch* nextBatch()
{
    if (gNextBatchIdx == 0 || gNextBatchIdx >= NUM_THREADS)
    {
        std::vector<std::thread> threads = { };
        threads.reserve(NUM_THREADS);

        for (size_t threadId = 0; threadId < NUM_THREADS; threadId++)
            threads.push_back(std::thread(loadBatch, threadId));

        // Wait for the threads
        for (auto& thread : threads)
            if (thread.joinable())
                thread.join();

        gDataFilePos += static_cast<size_t>(sizeof(DataEntry)) * BATCH_SIZE * NUM_THREADS;

        if (gDataFilePos >= DATA_FILE_BYTES)
            gDataFilePos -= DATA_FILE_BYTES;

        gNextBatchIdx = 0;
    }

    return &gBatches[gNextBatchIdx++];
}

int main()
{
    return 0;
}
