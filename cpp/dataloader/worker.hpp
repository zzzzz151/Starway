#pragma once

#include <fstream>
#include <future>

#include "../chess/move_gen.hpp"
#include "../chess/position.hpp"
#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../converter/data_entry.hpp"
#include "../utils.hpp"
#include "batch.hpp"
#include "move_mapping.hpp"

class Worker {
   private:
    std::ifstream mDataFile;
    size_t mFileSizeBytes;
    Batch mBatch;

   public:
    std::future<Batch*> mFuture;

    constexpr Worker(const size_t id,
                     const std::string& dataFilePath,
                     const size_t fileSizeBytes,
                     const size_t batchSize) {
        // Open data file and seek it to the start of this worker's first batch
        mDataFile = std::ifstream(dataFilePath, std::ios::binary);
        assert(mDataFile);
        mDataFile.seekg(static_cast<i64>(id * batchSize * sizeof(StarwayDataEntry)), std::ios::beg);

        mFileSizeBytes = fileSizeBytes;
        mBatch = Batch(batchSize);
    }

    constexpr Batch* getNextBatch(const size_t numWorkers, const size_t batchSize) {
        // Our data file's cursor is already at the start of the batch

        assert(mDataFile);

        const auto mirrorVAxis = [](const Square kingSq) -> bool {
            return static_cast<i32>(fileOf(kingSq)) < static_cast<i32>(File::E);
        };

        for (size_t entryIdx = 0; entryIdx < batchSize; entryIdx++) {
            StarwayDataEntry entry;

            mDataFile.read(reinterpret_cast<char*>(&entry), sizeof(entry));
            assert(mDataFile);

            entry.validate();

            Position pos;
            pos.reset();

            const bool inCheck = entry.get(Mask::IN_CHECK);

            const Square ourKingSqOriented =
                static_cast<Square>(entry.get(Mask::OUR_KING_SQ_ORIENTED));

            const Square theirKingSqOriented =
                static_cast<Square>(entry.get(Mask::THEIR_KING_SQ_ORIENTED));

            // Flip ranks if black to move
            // Flip files if that color's king is on left side of board
            const u8 stmXor = mirrorVAxis(ourKingSqOriented) ? 7 : 0;
            const u8 ntmXor = mirrorVAxis(theirKingSqOriented) ? 56 ^ 7 : 56;

            // Iterate pieces
            size_t piecesSeen = 0;
            while (entry.mOccupied > 0) {
                const Square sq = popLsb(entry.mOccupied);
                const u8 pieceColor = entry.mPieces & 0b1;
                const u8 pieceType = (entry.mPieces & 0b1110) >> 1;
                assert(pieceType <= static_cast<u8>(PieceType::King));

                const size_t idx = entryIdx * MAX_PIECES_PER_POS + piecesSeen;

                // clang-format off

                // Set stm feature index which was -1
                mBatch.activeFeaturesStm[idx]
                    = inCheck * 768
                    + static_cast<i16>(pieceColor) * 384
                    + static_cast<i16>(pieceType) * 64
                    + static_cast<i16>(static_cast<u8>(sq) ^ stmXor);

                // Set nstm feature index which was -1
                mBatch.activeFeaturesNtm[idx]
                    = inCheck * 768
                    + static_cast<i16>(!pieceColor) * 384
                    + static_cast<i16>(pieceType) * 64
                    + static_cast<i16>(static_cast<u8>(sq) ^ ntmXor);

                // clang-format on

                pos.togglePiece(
                    static_cast<Color>(pieceColor), static_cast<PieceType>(pieceType), sq);

                entry.mPieces >>= 4;  // Get the next 4 bits piece ready
                piecesSeen++;
            }

            const size_t idx = entryIdx * MAX_PIECES_PER_POS + piecesSeen;

            mBatch.activeFeaturesStm[idx] = mBatch.activeFeaturesNtm[idx] = -1;

            if (entry.get(Mask::CASTLING_KS)) {
                pos.enableCastlingRight(pos.mSideToMove, true);
            }

            if (entry.get(Mask::CASTLING_QS)) {
                pos.enableCastlingRight(pos.mSideToMove, false);
            }

            if (entry.get(Mask::EP_FILE) < 8) {
                const File epFile = static_cast<File>(entry.get(Mask::EP_FILE));
                pos.setEpSquare(toSquare(epFile, Rank::Rank6));
            }

            mBatch.stmScores[entryIdx] = entry.mStmScore;
            mBatch.stmResults[entryIdx] = static_cast<float>(entry.get(Mask::STM_RESULT)) / 2.0f;

            // Fill mBatch.legalMovesIdxs slice and mBatch.bestMoveIdx for this data entry

            const auto legalMoves = getLegalMoves(pos);
            assert(legalMoves.size() > 0 && legalMoves.size() <= MAX_MOVES_PER_POS);

            bool bestMoveFound = false;

            for (size_t i = 0; i < legalMoves.size(); i++) {
                const MontyformatMove move = legalMoves[i];

                const MontyformatMove moveOriented =
                    mirrorVAxis(ourKingSqOriented) ? move.filesFlipped() : move;

                mBatch.legalMovesIdxs[entryIdx * MAX_MOVES_PER_POS + i] =
                    static_cast<i16>(mapMoveIdx(moveOriented));

                if (move == MontyformatMove(entry.mBestMove)) {
                    mBatch.bestMoveIdx[entryIdx] = static_cast<u8>(i);
                    bestMoveFound = true;
                }
            }

            assert(bestMoveFound);

            for (size_t i = legalMoves.size(); i < MAX_MOVES_PER_POS; i++) {
                mBatch.legalMovesIdxs[entryIdx * MAX_MOVES_PER_POS + i] = -1;
            }
        }

        // Set data file cursor to start of this worker's next batch

        i64 filePos = mDataFile.tellg();
        assert(filePos > 0);

        // Minus 1 because we just parsed a batch
        filePos += (numWorkers - 1) * batchSize * sizeof(StarwayDataEntry);

        mDataFile.seekg(filePos % static_cast<i64>(mFileSizeBytes), std::ios::beg);

        return &mBatch;
    }
};  // class Worker
