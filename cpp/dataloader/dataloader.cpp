#include <cstring>
#include <iostream>
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

    std::cout << "Batches in data file: " << BATCH_OFFSETS.size() << std::endl;

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

    // Set all features to -1 which indicates no piece
    std::fill(
        batch.activeFeaturesStm, batch.activeFeaturesStm + BATCH_SIZE * MAX_PIECES_PER_POS, -1);

    std::fill(
        batch.activeFeaturesNtm, batch.activeFeaturesNtm + BATCH_SIZE * MAX_PIECES_PER_POS, -1);

    batch.totalLegalMoves = 0;

    const auto mirrorVAxis = [](const Square kingSq) -> bool {
        return static_cast<i32>(fileOf(kingSq)) < static_cast<i32>(File::E);
    };

    for (size_t entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++) {
        StarwayDataEntry dataEntry;

        // Read data entry
        dataFile.read(reinterpret_cast<char*>(&dataEntry),
                      sizeof(u32) + sizeof(u64) + sizeof(u128) + sizeof(i16));

        assert(dataFile);

        // Read visits distribution of this entry
        dataFile.read(reinterpret_cast<char*>(&dataEntry.visits), dataEntry.visitsBytesCount());
        assert(dataFile);

        assert(std::popcount(dataEntry.occupied) > 2 && std::popcount(dataEntry.occupied) <= 32);

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
        while (dataEntry.occupied > 0) {
            const Square sq = popLsb(dataEntry.occupied);
            const u8 pieceColor = dataEntry.pieces & 0b1;
            const u8 pieceType = (dataEntry.pieces & 0b1110) >> 1;
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

            dataEntry.pieces >>= 4;  // Get the next 4 bits piece ready
            piecesSeen++;
        }

        // In the batch, set score and WDL of this entry

        const u8 stmWdl = static_cast<u8>(dataEntry.get(Mask::WDL));
        assert(stmWdl <= 2);

        batch.stmScores[entryIdx] = dataEntry.stmScore;
        batch.stmWDLs[entryIdx] = static_cast<float>(stmWdl) / 2.0f;

        u32 visitsSum = 0;
        for (size_t i = 0; i < static_cast<size_t>(dataEntry.get(Mask::NUM_MOVES)); i++) {
            visitsSum += dataEntry.visits[i].visits;
        }

        for (size_t i = 0; i < static_cast<size_t>(dataEntry.get(Mask::NUM_MOVES)); i++) {
            const auto [moveU16, visitsU8] = dataEntry.visits[i];

            const MontyformatMove moveOriented = mirrorVAxis(ourKingSqOriented)
                                                     ? MontyformatMove(moveU16).filesFlipped()
                                                     : MontyformatMove(moveU16);

            const size_t moveIdx = mapMoveIdx(moveOriented);

            // Store tuple (entryIdx, moveIdx, visitsPercent)

            batch.legalMovesIdxsAndVisitsPercent[batch.totalLegalMoves * 3] =
                static_cast<float>(entryIdx);

            batch.legalMovesIdxsAndVisitsPercent[batch.totalLegalMoves * 3 + 1] =
                static_cast<float>(moveIdx);

            batch.legalMovesIdxsAndVisitsPercent[batch.totalLegalMoves * 3 + 2] =
                static_cast<float>(visitsU8) / static_cast<float>(visitsSum);

            batch.totalLegalMoves++;
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
    std::cout << "Dataloader main()" << std::endl;
    return 0;
}
