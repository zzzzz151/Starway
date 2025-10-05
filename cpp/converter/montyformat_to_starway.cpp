/*
Usage:
./montyformat_to_starway
    <montyformat file>
    <output data file>
    <batch size>
    <batches to output>
*/

// Montyformat docs:
// https://github.com/official-monty/montyformat/blob/main/src/value.rs
// https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <vector>

#include "../chess/move_gen.hpp"
#include "../chess/position.hpp"
#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../dataloader/move_mapping.hpp"
#include "../utils.hpp"
#include "compressed_board.hpp"
#include "data_entry.hpp"
#include "data_filter.hpp"

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::println(std::cerr,
                     "Usage: {} {} {} {} {}",
                     argv[0],
                     "<montyformat input file>",
                     "<output data file>",
                     "<batch size>",
                     "<batches to output>");

        return 1;
    }

    // Read program args
    const std::string mfFilePath = argv[1];
    const std::string outDataFilePath = argv[2];
    const size_t batchSize = std::stoull(argv[3]);
    const size_t targetNumBatches = std::stoull(argv[4]);

    // Print program args
    std::println("Input data file: {}", mfFilePath);
    std::println("Output data file: {}", outDataFilePath);
    std::println("Batch size: {} data entries", batchSize);
    std::println("Batches to output: {}", targetNumBatches);

    assert(batchSize > 0);
    assert(targetNumBatches > 0);

    // Open files
    std::ifstream mfFile(mfFilePath, std::ios::binary);
    std::ofstream outDataFile(outDataFilePath, std::ios::binary);

    assert(mfFile);
    assert(outDataFile);

    DataFilter dataFilter = DataFilter();

    size_t gameNum = 0;
    size_t entriesWritten = 0;
    size_t entriesSkipped = 0;

    const auto printProgress = [&]() {
        std::println("Total data entries written: {}", entriesWritten);
        std::println("Total data entries skipped: {}", entriesSkipped);
        dataFilter.printStats();
    };

    while (entriesWritten < targetNumBatches * batchSize) {
        // Read compressed board
        CompressedBoard compressedBoard;
        mfFile.read(reinterpret_cast<char*>(&compressedBoard), sizeof(compressedBoard));

        // End of the montyformat input file?
        if (!mfFile) {
            break;
        }

        // New game
        gameNum++;

        // Convert compressed board to our position class which is easier to work with
        Position pos = compressedBoard.decompress();
        pos.validate();

        // Read game result from white POV
        u8 mfWhiteResult;
        mfFile.read(reinterpret_cast<char*>(&mfWhiteResult), sizeof(mfWhiteResult));
        assert(mfFile);
        assert(mfWhiteResult <= 2);

        // Iterate the game's positions (1 pos = 1 Starway data entry)
        while (entriesWritten < targetNumBatches * batchSize) {
            MontyformatMove mfBestMove;
            i16 mfWhiteScore;

            // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#moves-and-their-associated-information
            mfFile.read(reinterpret_cast<char*>(&mfBestMove), sizeof(mfBestMove));
            assert(mfFile);

            mfFile.read(reinterpret_cast<char*>(&mfWhiteScore), sizeof(mfWhiteScore));
            assert(mfFile);

            // 4 zero bytes = game terminator
            if (mfBestMove.isNull()) {
                assert(mfWhiteScore == 0);
                break;
            }

            // Validate move
            const PieceType ptMoving = pos.pieceAt(mfBestMove.getSrc()).value().second;
            mfBestMove.validate(pos.mSideToMove == Color::White, ptMoving);
            assert(mapMoveIdx(mfBestMove.maybeRanksFlipped(pos.mSideToMove)) >= 0);

            const auto legalMoves = getLegalMoves(pos);
            assert(legalMoves.contains(mfBestMove));

            // If not filtered out, write data entry to output data file
            if (!dataFilter.shouldSkip(pos, mfWhiteScore, legalMoves.size())) {
                StarwayDataEntry entry;

                const u8 stmResult =
                    pos.mSideToMove == Color::White ? mfWhiteResult : 2 - mfWhiteResult;

                entry.setMiscData(pos, stmResult);
                entry.setOccAndPieces(pos);

                entry.mStmScore = pos.mSideToMove == Color::White ? mfWhiteScore
                                                                  : static_cast<i16>(-mfWhiteScore);

                entry.mBestMove =
                    MontyformatMove(mfBestMove).maybeRanksFlipped(pos.mSideToMove).asU16();

                entry.validate();

                outDataFile.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
                assert(outDataFile);

                entriesWritten++;
            } else {
                entriesSkipped++;
            }

            // Log conversion progress once in a while
            if (entriesWritten % 16'777'216 == 0) {
                std::println("\nCurrently on game #{}", gameNum);
                printProgress();
            }

            pos.makeMove(mfBestMove);
            pos.validate();
        }
    }

    std::println("\nFinished; parsed {} games", gameNum);
    printProgress();

    return 0;
}
