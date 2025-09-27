/*
Usage:
./montyformat_to_starway
    <montyformat file>
    <output data file>
    <max RAM usage in MB>
    <batch offsets output file>
    <batch size>
    <batches to output>
*/

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
#include "../utils.hpp"
#include "compressed_board.hpp"
#include "data_entry.hpp"
#include "data_filter.hpp"

// Returns number of entries written
constexpr size_t shuffleWriteClearBuffer(std::vector<StarwayDataEntry>& buffer,
                                         std::ofstream& outDataFile,
                                         std::ofstream& batchOffsetsOutFile,
                                         const size_t batchSize) {
    // No partial batches: if the trailing data entries are a partial batch, discard it
    buffer.resize(buffer.size() - buffer.size() % batchSize);
    assert(buffer.size() % batchSize == 0);

    // Shuffle data entries in buffer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(buffer.begin(), buffer.end(), gen);

    // Iterate data entries in buffer
    for (size_t i = 0; i < buffer.size(); i++) {
        StarwayDataEntry& entry = buffer[i];

        // If starting a new batch, write its offset to the batch offsets output file
        if (i % batchSize == 0) {
            const size_t batchOffset = static_cast<size_t>(outDataFile.tellp());

            batchOffsetsOutFile.write(reinterpret_cast<const char*>(&batchOffset),
                                      sizeof(batchOffset));
        }

        // Write data entry to output data file
        entry.writeToOut(outDataFile);
    }

    // clear() does not change vector's capacity
    const size_t bufferSize = buffer.size();
    buffer.clear();
    return bufferSize;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::println(std::cerr,
                     "Usage: {} {} {} {} {} {} {}",
                     argv[0],
                     "<montyformat input file>",
                     "<output data file>",
                     "<max RAM usage in MB>",
                     "<batch offsets output file>",
                     "<batch size>",
                     "<batches to output>");

        return 1;
    }

    // Read program args
    const std::string mfFilePath = argv[1];
    const std::string outDataFilePath = argv[2];
    const size_t maxRamMB = std::stoull(argv[3]);
    const std::string batchOffsetsOutFilePath = argv[4];
    const size_t batchSize = std::stoull(argv[5]);
    const size_t targetNumBatches = std::stoull(argv[6]);

    // Print program args
    std::println("Input data file: {}", mfFilePath);
    std::println("Output data file: {}", outDataFilePath);
    std::println("Max RAM usage in MB: {}", maxRamMB);
    std::println("Batch offsets output file: {}", batchOffsetsOutFilePath);
    std::println("Batch size: {} data entries", batchSize);
    std::println("Batches to output: {}", targetNumBatches);

    assert(maxRamMB > 0);
    assert(batchSize > 0);
    assert(targetNumBatches > 0);

    // Open files
    std::ifstream mfFile(mfFilePath, std::ios::binary);
    std::ofstream outDataFile(outDataFilePath, std::ios::binary);
    std::ofstream batchOffsetsOutFile(batchOffsetsOutFilePath, std::ios::binary);

    assert(mfFile);
    assert(outDataFile);
    assert(batchOffsetsOutFile);

    // How many StarwayDataEntry can the buffer hold?
    size_t bufferCapacity = maxRamMB * 1000 * 1000 / sizeof(StarwayDataEntry);
    bufferCapacity -= bufferCapacity % batchSize;

    std::println("Buffer capacity: {} data entries", bufferCapacity);
    assert(bufferCapacity >= batchSize);

    std::vector<StarwayDataEntry> buffer;
    buffer.reserve(bufferCapacity);

    DataFilter dataFilter = DataFilter();

    size_t gameNum = 0;
    size_t entriesWritten = 0;
    size_t entriesSkipped = 0;

    const auto printProgress = [&]() {
        assert(entriesWritten % batchSize == 0);
        std::println("Total batches written: {}", entriesWritten / batchSize);
        std::println("Total data entries written: {}", entriesWritten);
        std::println("Total data entries skipped: {}", entriesSkipped);
        dataFilter.printCounts();
    };

    while (entriesWritten + buffer.size() < targetNumBatches * batchSize) {
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
        // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#game-outcome
        u8 mfWhiteResult;
        mfFile.read(reinterpret_cast<char*>(&mfWhiteResult), sizeof(mfWhiteResult));
        assert(mfFile);
        assert(mfWhiteResult <= 2);

        const auto getStmResult = [&]() -> u8 {
            return pos.mSideToMove == Color::White ? mfWhiteResult : 2 - mfWhiteResult;
        };

        // Iterate the game's positions (1 pos = 1 Starway data entry)
        while (entriesWritten + buffer.size() < targetNumBatches * batchSize) {
            // We will read from the montyformat file into these 4 fields
            MontyformatMove mfBestMove;
            u16 mfScore;
            u8 mfMovesCount;
            std::array<u8, 218> visits;

            // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#moves-and-their-associated-information
            mfFile.read(reinterpret_cast<char*>(&mfBestMove), sizeof(mfBestMove));
            assert(mfFile);

            // Null move = game terminator
            // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#null-terminator
            if (mfBestMove.isNull()) {
                break;
            }

            // Validate move
            const PieceType ptMoving = pos.pieceAt(mfBestMove.getSrc()).value().second;
            mfBestMove.validate(pos.mSideToMove == Color::White, ptMoving);
            assert(mapMoveIdx(mfBestMove.maybeRanksFlipped(pos.mSideToMove)) >= 0);

            // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#score
            mfFile.read(reinterpret_cast<char*>(&mfScore), sizeof(mfScore));
            assert(mfFile);

            // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#move-count
            mfFile.read(reinterpret_cast<char*>(&mfMovesCount), sizeof(mfMovesCount));
            assert(mfFile);
            assert(mfMovesCount > 0 && static_cast<size_t>(mfMovesCount) <= visits.size());

            // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#visit-distribution
            mfFile.read(reinterpret_cast<char*>(&visits), mfMovesCount * sizeof(u8));
            assert(mfFile);

            const double stmScoreSigmoided =
                static_cast<double>(mfScore) / static_cast<double>(std::numeric_limits<u16>::max());

            auto legalMoves = getLegalMoves(pos);
            assert(static_cast<size_t>(mfMovesCount) == legalMoves.size());
            // assert(legalMoves.contains(mfBestMove));

            // Sort moves in ascending order since that's how visits in montyformat are ordered
            std::sort(legalMoves.begin(),
                      legalMoves.end(),
                      [](const MontyformatMove a, const MontyformatMove b) {
                          return a.asU16() < b.asU16();
                      });

            i32 bestMoveVisits = -1;
            for (size_t i = 0; i < legalMoves.size(); i++) {
                if (legalMoves[i] == mfBestMove) {
                    bestMoveVisits = visits[i];
                    break;
                }
            }

            assert(bestMoveVisits >= 0);

            // Filter out this data entry?
            if (dataFilter.shouldSkip(
                    pos, legalMoves.size(), stmScoreSigmoided, static_cast<u8>(bestMoveVisits))) {
                pos.makeMove(mfBestMove);
                pos.validate();

                entriesSkipped++;
                continue;
            }

            buffer.push_back(StarwayDataEntry());
            StarwayDataEntry& dataEntry = buffer.back();

            dataEntry.setMiscData(pos, getStmResult(), mfMovesCount);
            dataEntry.setOccAndPieces(pos);
            dataEntry.mStmScore = mfScore;

            dataEntry.validate();

            // Load visits distribution into StarwayDataEntry object
            u8 highestVisits = 0;
            for (size_t i = 0; i < legalMoves.size(); i++) {
                const MontyformatMove moveOriented =
                    legalMoves[i].maybeRanksFlipped(pos.mSideToMove);

                dataEntry.mVisits[i] = {.move = moveOriented.asU16(), .visits = visits[i]};

                highestVisits = std::max<u8>(visits[i], highestVisits);
            }

            assert(highestVisits == 255);

            // Data entries buffer is full?
            if (buffer.size() >= buffer.capacity()) {
                // Write data entries in buffer to output data file
                entriesWritten +=
                    shuffleWriteClearBuffer(buffer, outDataFile, batchOffsetsOutFile, batchSize);

                assert(entriesWritten % batchSize == 0);

                // Log conversion progress once in a while
                std::println("\nCurrently on game #{}", gameNum);
                printProgress();
            }

            pos.makeMove(mfBestMove);
            pos.validate();
        }
    }

    // Write leftover data entries that are still in the buffer
    entriesWritten += shuffleWriteClearBuffer(buffer, outDataFile, batchOffsetsOutFile, batchSize);
    assert(entriesWritten % batchSize == 0);

    std::println("\nFinished; parsed {} games", gameNum);
    printProgress();

    // Number of batch offsets equals number of batches
    assert(static_cast<size_t>(batchOffsetsOutFile.tellp()) / sizeof(size_t) ==
           entriesWritten / batchSize);

    // Print interleave command
    std::println("\nRun ./interleave {} {} {} {} {} {}",
                 outDataFilePath,
                 "interleaved_" + outDataFilePath,
                 bufferCapacity,
                 batchOffsetsOutFilePath,
                 "interleaved_" + batchOffsetsOutFilePath,
                 batchSize);

    return 0;
}
