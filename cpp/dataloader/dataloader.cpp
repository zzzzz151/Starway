#include <cstring>
#include <iostream>
#include <print>
#include <thread>
#include <vector>

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
std::vector<size_t> BATCH_OFFSETS;
size_t BATCH_SIZE = 1;
size_t NUM_THREADS = 1;

std::vector<Batch> gBatches = {};  // NUM_THREADS batches
size_t gTotalBatchesYielded = 0;

extern "C" API void init(const char* dataFilePath,
                         const char* batchOffsetsFilePath,
                         const size_t batchSize,
                         const size_t numThreads) {
    assert(batchSize > 0);
    assert(numThreads > 0);

    DATA_FILE_PATH = dataFilePath;
    BATCH_SIZE = batchSize;
    NUM_THREADS = numThreads;

    // Open batch offsets file at its end
    std::ifstream batchOffsetsFile(static_cast<std::string>(batchOffsetsFilePath),
                                   std::ios::binary | std::ios::ate);

    assert(batchOffsetsFile);

    // Move batch offsets from batch offsets file into RAM

    BATCH_OFFSETS.resize(static_cast<size_t>(batchOffsetsFile.tellg()) / sizeof(size_t));
    assert(BATCH_OFFSETS.size() > 0);

    std::println("Batches in data file: {}", BATCH_OFFSETS.size());

    batchOffsetsFile.seekg(0, std::ios::beg);

    batchOffsetsFile.read(reinterpret_cast<char*>(BATCH_OFFSETS.data()),
                          static_cast<i64>(BATCH_OFFSETS.size() * sizeof(size_t)));

    assert(batchOffsetsFile);

    // Allocate batches (1 for each thread)
    for (size_t i = 0; i < NUM_THREADS; i++) {
        gBatches.push_back(Batch(BATCH_SIZE));
    }
}

constexpr void loadBatch(const size_t threadId) {
    // Open data file
    std::ifstream dataFile(DATA_FILE_PATH, std::ios::binary);
    assert(dataFile);

    // In the data file, go to the position of our batch to read
    size_t idx = (gTotalBatchesYielded + threadId) % BATCH_OFFSETS.size();
    dataFile.seekg(static_cast<i64>(BATCH_OFFSETS[idx]), std::ios::beg);

    // Batch to fill
    Batch& batch = gBatches[threadId];

    const auto mirrorVAxis = [](const Square kingSq) -> bool {
        return static_cast<i32>(fileOf(kingSq)) < static_cast<i32>(File::E);
    };

    for (size_t entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++) {
        // Read from data file to StarwayDataEntry object
        StarwayDataEntry dataEntry = StarwayDataEntry(dataFile);
        // dataEntry.validate();

        const bool inCheck = dataEntry.get(Mask::IN_CHECK);

        const Square ourKingSqOriented =
            static_cast<Square>(dataEntry.get(Mask::OUR_KING_SQ_ORIENTED));

        const Square theirKingSqOriented =
            static_cast<Square>(dataEntry.get(Mask::THEIR_KING_SQ_ORIENTED));

        // Flip ranks if black to move
        // Flip files if that color's king is on left side of board
        const u8 stmXor = mirrorVAxis(ourKingSqOriented) ? 7 : 0;
        const u8 ntmXor = mirrorVAxis(theirKingSqOriented) ? 56 ^ 7 : 56;

        // Iterate pieces
        size_t piecesSeen = 0;
        while (dataEntry.mOccupied > 0) {
            const Square sq = popLsb(dataEntry.mOccupied);
            const u8 pieceColor = dataEntry.mPieces & 0b1;
            const u8 pieceType = (dataEntry.mPieces & 0b1110) >> 1;
            assert(pieceType <= static_cast<u8>(PieceType::King));

            // Index of this feature in the batch's array
            idx = entryIdx * MAX_PIECES_PER_POS + piecesSeen;

            // clang-format off

            // Set stm feature index which was -1
            batch.activeFeaturesStm[idx]
                = inCheck * 768
                + static_cast<i16>(pieceColor) * 384
                + static_cast<i16>(pieceType) * 64
                + static_cast<i16>(static_cast<u8>(sq) ^ stmXor);

            // Set nstm feature index which was -1
            batch.activeFeaturesNtm[idx]
                = inCheck * 768
                + static_cast<i16>(!pieceColor) * 384
                + static_cast<i16>(pieceType) * 64
                + static_cast<i16>(static_cast<u8>(sq) ^ ntmXor);

            // clang-format on

            dataEntry.mPieces >>= 4;  // Get the next 4 bits piece ready
            piecesSeen++;
        }

        idx = entryIdx * MAX_PIECES_PER_POS + piecesSeen;

        batch.activeFeaturesStm[idx] = batch.activeFeaturesNtm[idx] = -1;

        batch.stmScoresSigmoided[entryIdx] = static_cast<float>(dataEntry.mStmScore) /
                                             static_cast<float>(std::numeric_limits<u16>::max());

        batch.stmResults[entryIdx] = static_cast<float>(dataEntry.get(Mask::STM_RESULT)) / 2.0f;

        const size_t numMoves = static_cast<size_t>(dataEntry.get(Mask::NUM_MOVES));

        u32 visitsSum = 0;
        for (size_t i = 0; i < numMoves; i++) {
            visitsSum += dataEntry.mVisits[i].visits;
        }

        for (size_t i = 0; i < numMoves; i++) {
            const auto [moveU16, visitsU8] = dataEntry.mVisits[i];

            const MontyformatMove moveOriented = mirrorVAxis(ourKingSqOriented)
                                                     ? MontyformatMove(moveU16).filesFlipped()
                                                     : MontyformatMove(moveU16);

            idx = entryIdx * MAX_MOVES_PER_POS + i;

            batch.legalMovesIdxs[idx] = static_cast<i16>(mapMoveIdx(moveOriented));
            batch.visitsPercent[idx] = static_cast<float>(visitsU8) / static_cast<float>(visitsSum);
        }

        for (size_t i = numMoves; i < MAX_MOVES_PER_POS; i++) {
            idx = entryIdx * MAX_MOVES_PER_POS + i;

            batch.legalMovesIdxs[idx] = -1;
            batch.visitsPercent[idx] = 0.0f;
        }
    }
}

// Gets the next batch for pytorch
// When no batch is ready, we launch NUM_THREADS threads and each generates 1 batch
// When those NUM_THREADS batches have been yielded, we generate NUM_THREADS batches again
extern "C" API Batch* nextBatch() {
    if (gTotalBatchesYielded % NUM_THREADS == 0) {
        std::vector<std::thread> threads = {};
        threads.reserve(NUM_THREADS);

        for (size_t threadId = 0; threadId < NUM_THREADS; threadId++) {
            threads.push_back(std::thread(loadBatch, threadId));
        }

        // Wait for the threads
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    Batch& batch = gBatches[gTotalBatchesYielded % NUM_THREADS];
    gTotalBatchesYielded++;
    return &batch;
}

int main() {
    std::println("Dataloader main()");
    return 0;
}
