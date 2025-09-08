// clang-format off

#include "utils.hpp"
#include "move.hpp"
#include "data_entry.hpp"
#include "batch.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wold-style-cast"

#include "chess.hpp"

#pragma GCC diagnostic pop

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <thread>
#include <cstring>

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

std::vector<Batch> gBatches = { }; // NUM_THREADS batches
size_t gNextBatchIdx = 0; // 0 to NUM_THREADS-1
size_t gDataFilePos = 0;

extern "C" API void init(
    const char* dataFilePath, const i32 batchSize, const i32 numThreads)
{
    DATA_FILE_PATH = static_cast<std::string>(dataFilePath);

    // Open file in binary mode and at the end
    std::ifstream dataFile(DATA_FILE_PATH, std::ios::binary | std::ios::ate);

    if (!dataFile || !dataFile.is_open()) {
        std::cout << "Error opening file " << DATA_FILE_PATH << std::endl;
        exit(EXIT_FAILURE);
    }

    if (batchSize <= 0) {
        std::cout << "Batch size must be > 0 but is " << batchSize << std::endl;
        exit(EXIT_FAILURE);
    }

    if (numThreads <= 0) {
        std::cout << "Threads count must be > 0 but is " << numThreads << std::endl;
        exit(EXIT_FAILURE);
    }

    BATCH_SIZE = static_cast<size_t>(batchSize);
    NUM_THREADS = static_cast<size_t>(numThreads);

    DATA_FILE_BYTES = static_cast<size_t>(dataFile.tellg());

    if (DATA_FILE_BYTES % static_cast<size_t>(sizeof(DataEntry)) != 0) {
        std::cout << "Data file bytes must divide data entry bytes but it doesn't" << std::endl;
        exit(EXIT_FAILURE);
    }

    const size_t numDataEntries = DATA_FILE_BYTES / static_cast<size_t>(sizeof(DataEntry));

    if (numDataEntries % BATCH_SIZE != 0) {
        std::cout << "Data entries count must divide batch size but it doesn't" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < NUM_THREADS; i++)
        gBatches.push_back(Batch(BATCH_SIZE));
}

inline void loadBatch(const size_t threadId) {
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
    batch.totalLegalMoves = 0;

    std::fill(batch.activeFeaturesStm, batch.activeFeaturesStm + BATCH_SIZE * 32, -1);
    std::fill(batch.activeFeaturesNtm, batch.activeFeaturesNtm + BATCH_SIZE * 32, -1);

    DataEntry dataEntry;
    chess::Board board;

    for (size_t entryIdx = 0; entryIdx < BATCH_SIZE; entryIdx++) {
        dataFile.read(reinterpret_cast<char*>(&dataEntry), sizeof(DataEntry));
        board.reset();

        std::optional<std::tuple<u8, u8, u8>> pieceData = std::nullopt;
        size_t piecesSeen = 0;

        const u8 stmXor = dataEntry.stmKingSqOriented() % 8 <= 3 ? 7 : 0;
        const u8 ntmXor = dataEntry.ntmKingSqOriented() % 8 <= 3 ? 56 ^ 7 : 56;

        while ((pieceData = dataEntry.popOrientedPiece()).has_value()) {
            const auto [pieceColor, pieceType, square] = *pieceData;

            const size_t idx = entryIdx * 32 + piecesSeen;

            batch.activeFeaturesStm[idx]
                = dataEntry.inCheck() * 768
                + static_cast<i16>(pieceColor) * 384
                + static_cast<i16>(pieceType) * 64
                + static_cast<i16>(square ^ stmXor);

            batch.activeFeaturesNtm[idx]
                = dataEntry.inCheck() * 768
                + static_cast<i16>(!pieceColor) * 384
                + static_cast<i16>(pieceType) * 64
                + static_cast<i16>(square ^ ntmXor);

            batch.numActiveFeatures++;

            chess::Piece piece = chess::Piece(
                static_cast<chess::PieceType::underlying>(pieceType),
                chess::Color(static_cast<i8>(pieceColor))
            );

            board.placePiece(piece, chess::Square(square));
            piecesSeen++;
        }

        batch.stmScores[entryIdx] = dataEntry.stmScore();
        batch.stmWDLs[entryIdx] = dataEntry.stmWdl();

        batch.bestMoveIdx1882[entryIdx] = getMoveIdx1882(
            moveSrc(dataEntry.bestMoveOriented()) ^ stmXor,
            moveDst(dataEntry.bestMoveOriented()) ^ stmXor,
            getPieceTypeMoving(dataEntry.bestMoveOriented()),
            getPromotionPieceType(dataEntry.bestMoveOriented())
        );

        if (dataEntry.kCastleRightOriented()) {
            board.cr_.setCastlingRight(
                chess::Color::WHITE,
                chess::Board::CastlingRights::Side::KING_SIDE,
                chess::File::FILE_H
            );
        }

        if (dataEntry.qCastleRightOriented()) {
            board.cr_.setCastlingRight(
                chess::Color::WHITE,
                chess::Board::CastlingRights::Side::QUEEN_SIDE,
                chess::File::FILE_A
            );
        }

        board.init_castling_path();

        board.ep_sq_ = chess::Square(dataEntry.epSquareOriented());

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        // Train data only has positions with more than 0 legal moves and less than 128
        assert(moves.size() > 0 && moves.size() < 128);

        bool bestMoveInLegals = false;

        for (const auto& move : moves) {
            const i16 moveIdx1882 = getMoveIdx1882(
                static_cast<u8>(move.from().index()) ^ stmXor,
                static_cast<u8>(move.to().index()) ^ stmXor,
                static_cast<u8>(board.at(move.from()).type()),
                move.typeOf() == chess::Move::PROMOTION ? static_cast<u8>(move.promotionType()) : 6
            );

            if (moveIdx1882 == batch.bestMoveIdx1882[entryIdx])
                bestMoveInLegals = true;

            batch.legalMovesIdxs1882[batch.totalLegalMoves * 2] = static_cast<i16>(entryIdx);
            batch.legalMovesIdxs1882[batch.totalLegalMoves * 2 + 1] = moveIdx1882;

            batch.totalLegalMoves++;
        }

        assert(bestMoveInLegals);
    }
}

extern "C" API Batch* nextBatch() {
    if (gNextBatchIdx == 0 || gNextBatchIdx >= NUM_THREADS) {
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

int main() {
    std::cout << "main()" << std::endl;
    return 0;
}
