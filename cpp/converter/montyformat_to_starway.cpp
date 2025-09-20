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
#include <random>
#include <vector>

#include "../chess/move_gen.hpp"
#include "../chess/position.hpp"
#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../utils.hpp"
#include "compressed_board.hpp"
#include "data_entry.hpp"

constexpr u16 MIN_FULLMOVE_COUNTER = 9;
constexpr u8 MAX_HALFMOVE_CLOCK = 89;
constexpr i16 MAX_SCORE_CP = 10'000;

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
        outDataFile.write(reinterpret_cast<const char*>(&entry.mMiscData), sizeof(entry.mMiscData));
        outDataFile.write(reinterpret_cast<const char*>(&entry.mOccupied), sizeof(entry.mOccupied));
        outDataFile.write(reinterpret_cast<const char*>(&entry.mPieces), sizeof(entry.mPieces));
        outDataFile.write(reinterpret_cast<const char*>(&entry.mStmScore), sizeof(entry.mStmScore));

        // For the visits array, we only write the filled elements (number of legal moves)
        outDataFile.write(reinterpret_cast<const char*>(&entry.mVisits),
                          static_cast<i64>(entry.visitsBytesCount()));
    }

    // clear() does not change vector's capacity
    const size_t bufferSize = buffer.size();
    buffer.clear();
    return bufferSize;
}

// https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#score
constexpr i16 mfScoreToCentipawns(const u16 mfScore) {
    double wdl =
        static_cast<double>(mfScore) / static_cast<double>(std::numeric_limits<u16>::max());

    if (wdl <= 0.0) {
        return -32767;
    }

    if (wdl >= 1.0) {
        return 32767;
    }

    wdl *= 2.0;
    wdl -= 1.0;

    const i64 centipawns = llround(660.6 * wdl / (1 - 0.9751875 * std::pow(wdl, 10)));
    return static_cast<i16>(std::clamp<i64>(centipawns, -32767, 32767));
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: ";
        std::cerr << argv[0];
        std::cerr << " <montyformat input file>";
        std::cerr << " <output data file>";
        std::cerr << " <max RAM usage in MB>";
        std::cerr << " <batch offsets output file>";
        std::cerr << " <batch size>";
        std::cerr << " <batches to output>";
        std::cerr << std::endl;
        return 1;
    }

    // Read program args
    const std::string mfFileName = argv[1];
    const std::string outDataFileName = argv[2];
    const size_t maxRamMB = std::stoull(argv[3]);
    const std::string batchOffsetsOutFileName = argv[4];
    const size_t batchSize = std::stoull(argv[5]);
    const size_t targetNumBatches = std::stoull(argv[6]);

    // Print program args
    std::cout << "Input data file: " << mfFileName << std::endl;
    std::cout << "Output data file: " << outDataFileName << std::endl;
    std::cout << "Max RAM usage in MB: " << maxRamMB << std::endl;
    std::cout << "Batch offsets output file: " << batchOffsetsOutFileName << std::endl;
    std::cout << "Batch size: " << batchSize << " data entries" << std::endl;
    std::cout << "Batches to output: " << targetNumBatches << std::endl;

    assert(maxRamMB > 0);
    assert(batchSize > 0);
    assert(targetNumBatches > 0);

    // Open files
    std::ifstream mfFile(mfFileName, std::ios::binary);
    std::ofstream outDataFile(outDataFileName, std::ios::binary);
    std::ofstream batchOffsetsOutFile(batchOffsetsOutFileName, std::ios::binary);

    assert(mfFile);
    assert(outDataFile);
    assert(batchOffsetsOutFile);

    // How many StarwayDataEntry can the buffer hold?
    size_t bufferCapacity = maxRamMB * 1000 * 1000 / sizeof(StarwayDataEntry);
    bufferCapacity -= bufferCapacity % batchSize;

    std::cout << "Buffer capacity: " << bufferCapacity << " data entries" << std::endl;
    assert(bufferCapacity >= batchSize);

    std::vector<StarwayDataEntry> buffer;
    buffer.reserve(bufferCapacity);

    size_t gameNum = 0;
    size_t entriesWritten = 0;
    size_t entriesSkipped = 0;

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

        // Read white WDL
        // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#game-outcome
        u8 mfWhiteWdl;
        mfFile.read(reinterpret_cast<char*>(&mfWhiteWdl), sizeof(mfWhiteWdl));
        assert(mfFile);
        assert(mfWhiteWdl <= 2);

        const auto getStmWdl = [&]() -> u8 {
            return pos.mSideToMove == Color::White ? mfWhiteWdl : 2 - mfWhiteWdl;
        };

        // Iterate the game's positions (1 pos = 1 Starway data entry)
        while (entriesWritten + buffer.size() < targetNumBatches * batchSize) {
            // We will read from the montyformat file into these 3 fields
            MontyformatMove mfBestMove;
            u16 mfScore;
            u8 mfMovesCount;

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
            assert(mfMovesCount > 0 && mfMovesCount <= 218);

            const i16 stmScoreCp = mfScoreToCentipawns(mfScore);

            // Data filtering
            bool skip = pos.isInsufficientMaterial();
            skip |= pos.getFullMoveCounter() < MIN_FULLMOVE_COUNTER;
            skip |= pos.getHalfMoveClock() > MAX_HALFMOVE_CLOCK;
            skip |= std::abs(stmScoreCp) > MAX_SCORE_CP;

            if (skip) {
                // Skip visits distribution
                mfFile.seekg(sizeof(u8) * mfMovesCount, std::ios::cur);
                assert(mfFile);

                pos.makeMove(mfBestMove);
                pos.validate();

                entriesSkipped++;
                continue;
            }

            buffer.push_back(StarwayDataEntry());
            StarwayDataEntry& dataEntry = buffer.back();

            dataEntry.setMiscData(pos, getStmWdl(), mfMovesCount);
            dataEntry.setOccAndPieces(pos);
            dataEntry.mStmScore = stmScoreCp;

            dataEntry.validate();

            auto legalMoves = getLegalMoves(pos);
            assert(static_cast<size_t>(mfMovesCount) == legalMoves.size());
            assert(legalMoves.contains(mfBestMove));

            // Sort moves in ascending order since that's how visits in montyformat are ordered
            std::sort(legalMoves.begin(),
                      legalMoves.end(),
                      [](const MontyformatMove a, const MontyformatMove b) {
                          return a.asU16() < b.asU16();
                      });

            // Read visits distribution (looping the legal moves)
            // https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#visit-distribution
            u8 highestVisits = 0;
            for (size_t i = 0; i < mfMovesCount; i++) {
                u8 visits;
                mfFile.read(reinterpret_cast<char*>(&visits), sizeof(visits));
                assert(mfFile);

                const MontyformatMove moveOriented =
                    legalMoves[i].maybeRanksFlipped(pos.mSideToMove);

                dataEntry.mVisits[i] = {.move = moveOriented.asU16(), .visits = visits};

                highestVisits = std::max<u8>(visits, highestVisits);
            }

            assert(highestVisits == 255);

            // Data entries buffer is full?
            if (buffer.size() >= buffer.capacity()) {
                // Write data entries in buffer to output data file
                entriesWritten +=
                    shuffleWriteClearBuffer(buffer, outDataFile, batchOffsetsOutFile, batchSize);

                assert(entriesWritten % batchSize == 0);

                // Log conversion progress once in a while
                const size_t batchesWritten = entriesWritten / batchSize;
                std::cout << "\nCurrently on game #" << gameNum << std::endl;
                std::cout << "Total batches written: " << batchesWritten << std::endl;
                std::cout << "Total data entries written: " << entriesWritten << std::endl;
                std::cout << "Total data entries skipped: " << entriesSkipped << std::endl;
            }

            pos.makeMove(mfBestMove);
            pos.validate();
        }
    }

    // Write leftover data entries that are still in the buffer
    entriesWritten += shuffleWriteClearBuffer(buffer, outDataFile, batchOffsetsOutFile, batchSize);
    assert(entriesWritten % batchSize == 0);

    const size_t batchesWritten = entriesWritten / batchSize;
    std::cout << "\nParsed " << gameNum << " games" << std::endl;
    std::cout << "Total batches written: " << batchesWritten << std::endl;
    std::cout << "Total data entries written: " << entriesWritten << std::endl;
    std::cout << "Total data entries skipped: " << entriesSkipped << std::endl;

    assert(static_cast<size_t>(batchOffsetsOutFile.tellp()) / sizeof(size_t) == batchesWritten);

    // Print interleave command
    std::cout << "\nRun ./interleave";
    std::cout << " " << outDataFileName;
    std::cout << " interleaved_" << outDataFileName;
    std::cout << " " << bufferCapacity;
    std::cout << " " << batchOffsetsOutFileName;
    std::cout << " interleaved_" << batchOffsetsOutFileName;
    std::cout << " " << batchSize;
    std::cout << std::endl;

    return 0;
}
