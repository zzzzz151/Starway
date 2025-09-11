#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>

#include "../chess/move_gen.hpp"
#include "../chess/position.hpp"
#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../utils.hpp"
#include "compressed_board.hpp"
#include "data_entry.hpp"

constexpr u16 MIN_FULLMOVE_COUNTER = 9;
constexpr u8 MAX_HALFMOVE_CLOCK = 89;
constexpr i32 MAX_SCORE_CP = 8000;

constexpr i16 mfScoreToCentipawns(const u16 mfScore) {
    const double wdl =
        static_cast<double>(mfScore) / static_cast<double>(std::numeric_limits<u16>::max());

    if (wdl == 0.0) {
        return -32767;
    }

    if (wdl == 1.0) {
        return 32767;
    }

    const double unsigmoided = std::log(wdl / (1.0 - wdl)) * 400.0;
    const i32 cp = static_cast<i32>(round(unsigmoided));

    return static_cast<i16>(std::clamp<i32>(cp, -32767, 32767));
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <montyformat_file> <data_entries_limit>"
                  << std::endl;

        return 1;
    }

    const std::string mfFileName = argv[1];
    std::cout << "Input file: " << mfFileName << std::endl;

    std::ifstream mfFile(mfFileName, std::ios::binary);
    if (!mfFile) {
        std::cerr << "Error: Could not open file " << mfFileName << std::endl;
        return 1;
    }

    const std::string outFileName = "converted.bin";
    std::cout << "Output file: " << outFileName << std::endl;

    std::ofstream outFile(outFileName, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << outFileName << std::endl;
        return 1;
    }

    const std::optional<size_t> dataEntriesLimit =
        argc > 2 ? std::optional<size_t>(std::stoul(argv[2])) : std::nullopt;

    std::cout << "Data entries limit: "
              << (dataEntriesLimit.has_value() ? std::to_string(*dataEntriesLimit) : "none")
              << std::endl;

    size_t gameNum = 0;
    size_t entriesWritten = 0;
    size_t entriesSkipped = 0;

    StarwayDataEntry dataEntry;

    while (mfFile && (!dataEntriesLimit.has_value() || entriesWritten < *dataEntriesLimit)) {
        CompressedBoard compressedBoard;
        mfFile.read(reinterpret_cast<char*>(&compressedBoard), sizeof(compressedBoard));

        if (!mfFile) {
            break;
        }

        gameNum++;
        // std::cout << "Reading game #" << gameNum << std::endl;

        Position pos = compressedBoard.decompress();
        pos.validate();

        u8 mfWhiteWdl;
        mfFile.read(reinterpret_cast<char*>(&mfWhiteWdl), sizeof(mfWhiteWdl));
        assert(mfFile);
        assert(mfWhiteWdl == 0 || mfWhiteWdl == 1 || mfWhiteWdl == 2);

        [[maybe_unused]] size_t posNum = 0;
        while (!dataEntriesLimit.has_value() || entriesWritten < *dataEntriesLimit) {
            posNum++;
            // std::cout << "Reading game #" << gameNum << " position #" << posNum << std::endl;

            MontyformatMove mfBestMove;
            u16 mfScore;
            u8 mfMovesCount;

            mfFile.read(reinterpret_cast<char*>(&mfBestMove), sizeof(mfBestMove));
            assert(mfFile);

            // Null move = game terminator
            if (mfBestMove.isNull()) {
                break;
            }

            mfBestMove.validate(pos.mSideToMove == Color::White);

            mfFile.read(reinterpret_cast<char*>(&mfScore), sizeof(mfScore));
            assert(mfFile);

            mfFile.read(reinterpret_cast<char*>(&mfMovesCount), sizeof(mfMovesCount));
            assert(mfFile);
            assert(mfMovesCount > 0 && mfMovesCount <= 218);

            dataEntry.setMiscData(
                pos, pos.mSideToMove == Color::White ? mfWhiteWdl : 2 - mfWhiteWdl, mfMovesCount);

            dataEntry.setOccAndPieces(pos);

            dataEntry.stmScore = mfScoreToCentipawns(mfScore);

            auto legalMoves = getLegalMoves(pos);

            assert(static_cast<size_t>(mfMovesCount) == legalMoves.size());
            assert(legalMoves.contains(mfBestMove));

            std::sort(legalMoves.begin(), legalMoves.end(),
                      [](const MontyformatMove a, const MontyformatMove b) {
                          return a.asU16() < b.asU16();
                      });

            // Read visits distribution
            u8 highestVisits = 0;
            for (size_t i = 0; i < mfMovesCount; i++) {
                u8 visits;
                mfFile.read(reinterpret_cast<char*>(&visits), sizeof(visits));
                assert(mfFile);

                const MontyformatMove moveOriented =
                    legalMoves[i].maybeRanksFlipped(pos.mSideToMove);

                dataEntry.visits[i] = MoveAndVisits{.move = moveOriented.asU16(), .visits = visits};

                highestVisits = std::max<u8>(visits, highestVisits);
            }

            assert(highestVisits == 255);

            const auto numPieces = std::popcount(pos.getOcc());

            const bool hasKnightOrBishop =
                (pos.getBb(PieceType::Knight) | pos.getBb(PieceType::Bishop)) > 0;

            bool skip = numPieces <= 2 || (numPieces == 3 && hasKnightOrBishop);
            skip |= pos.getFullMoveCounter() < MIN_FULLMOVE_COUNTER;
            skip |= pos.getHalfMoveClock() > MAX_HALFMOVE_CLOCK;
            skip |= std::abs(dataEntry.stmScore) > MAX_SCORE_CP;

            if (!skip) {
                dataEntry.writeToFile(outFile);
                entriesWritten++;
            } else {
                entriesSkipped++;
            }

            pos.makeMove(mfBestMove);
            pos.validate();

            if (!skip && entriesWritten % 1'048'576 == 0) {
                std::cout << "\nCurrently on game #" << gameNum << std::endl;
                std::cout << "Wrote " << entriesWritten << " data entries total" << std::endl;
                std::cout << "Skipped " << entriesSkipped << " data entries total" << std::endl;
            }
        }
    }

    std::cout << "\nFinished" << std::endl;
    std::cout << "Parsed " << gameNum << " games" << std::endl;
    std::cout << "Wrote " << entriesWritten << " data entries total" << std::endl;
    std::cout << "Skipped " << entriesSkipped << " data entries total" << std::endl;

    return 0;
}
