#include <cstring>
#include <iostream>
#include <thread>

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
size_t gTotalBatchesYielded = 0;

extern "C" API void init(const char* dataFilePath, const i32 batchSize, const i32 numThreads) {
    std::cout << "Batches in batch positions file: " << BATCH_POSITIONS.size() << std::endl;

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

    // Allocate gBatches
    for (size_t i = 0; i < NUM_THREADS; i++) {
        gBatches.push_back(Batch(BATCH_SIZE));
    }
}

constexpr void loadBatch(const size_t threadId) {
    // Open data file
    std::ifstream dataFile(DATA_FILE_PATH, std::ios::binary);
    assert(dataFile);

    // In the data file, go to the position of our batch to read
    size_t idx = (gTotalBatchesYielded + threadId) % BATCH_POSITIONS.size();
    dataFile.seekg(static_cast<i64>(BATCH_POSITIONS[idx]), std::ios::beg);

    // Fill the batch gBatches[threadId]

    Batch& batch = gBatches[threadId];
    batch.totalLegalMoves = 0;

    // Set all features to -1 which indicates no piece
    std::fill(batch.activeFeaturesStm, batch.activeFeaturesStm + BATCH_SIZE * MAX_PIECES, -1);
    std::fill(batch.activeFeaturesNtm, batch.activeFeaturesNtm + BATCH_SIZE * MAX_PIECES, -1);

    batch.totalLegalMoves = 0;

    // Zero all target logits
    std::memset(batch.logits, 0, sizeof(u8) * BATCH_SIZE * POLICY_OUTPUT_SIZE);

    StarwayDataEntry dataEntry;

    for (size_t entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++) {
        // Read data entry (position)
        dataFile.read(reinterpret_cast<char*>(&dataEntry),
                      sizeof(u32) + sizeof(u64) + sizeof(u128) + sizeof(i16));

        assert(dataFile);

        // Read visits distribution of this entry
        dataFile.read(reinterpret_cast<char*>(&dataEntry.visits), dataEntry.visitsBytesCount());
        assert(dataFile);

        const bool inCheck = dataEntry.get(Mask::IN_CHECK);

        const u8 ourKingSqOriented = static_cast<u8>(dataEntry.get(Mask::OUR_KING_SQ_ORIENTED));
        const u8 theirKingSqOriented = static_cast<u8>(dataEntry.get(Mask::THEIR_KING_SQ_ORIENTED));

        assert(ourKingSqOriented < 64 && theirKingSqOriented < 64);

        // Flip ranks if black to move
        // Flip files if that color's king is on left side of board
        const u8 stmXor = ourKingSqOriented % 8 <= 3 ? 7 : 0;
        const u8 ntmXor = theirKingSqOriented % 8 <= 3 ? 56 ^ 7 : 56;

        // Iterate pieces
        size_t piecesSeen = 0;
        while (dataEntry.occupied > 0) {
            const Square sq = popLsb(dataEntry.occupied);
            const u8 pieceColor = dataEntry.pieces & 0b1;
            const u8 pieceType = (dataEntry.pieces & 0b1110) >> 1;
            assert(pieceType <= static_cast<u8>(PieceType::King));

            // Index of this feature in the batch's array
            idx = entryIdx * MAX_PIECES + piecesSeen;

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
        assert(stmWdl == 0 || stmWdl == 1 || stmWdl == 2);

        batch.stmScores[entryIdx] = dataEntry.stmScore;
        batch.stmWDLs[entryIdx] = static_cast<float>(stmWdl) / 2.0f;

        // In the batch, set the legal moves indexes (output layer)
        // and set the target logits which is just the visits distribution of this position
        for (size_t i = 0; i < static_cast<size_t>(dataEntry.get(Mask::NUM_MOVES)); i++) {
            const auto [moveU16, visitsU8] = dataEntry.visits[i];
            const size_t moveIdx = mapMoveIdx(MontyformatMove(moveU16));

            // legalMovesIdx stores pairs (entryIdx, moveIdx) sequentially
            batch.legalMovesIdx[batch.totalLegalMoves * 2] = entryIdx;
            batch.legalMovesIdx[batch.totalLegalMoves * 2 + 1] = moveIdx;
            batch.totalLegalMoves++;

            // Careful to not overwrite the logits of other data entries!
            batch.logits[entryIdx * POLICY_OUTPUT_SIZE + moveIdx] = visitsU8;
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
