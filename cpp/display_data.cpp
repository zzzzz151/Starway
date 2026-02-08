/*
Usage:
./display_data
    <data file in Starway format>
    <data entry number from 1>
*/

#include <fstream>
#include <iostream>
#include <limits>
#include <print>

#include "chess/types.hpp"
#include "chess/util.hpp"
#include "converter/data_entry.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::println(std::cerr,
                     "Usage: {} {} {}",
                     argv[0],
                     "<data file in Starway format>",
                     "<data entry number from 1>");

        return 1;
    }

    // Read program args
    const std::string dataFilePath = argv[1];
    const size_t dataEntryNum = std::stoull(argv[2]);

    // Print program args
    std::println("Data file: {}", dataFilePath);
    std::println("Data entry number from 1: {}", dataEntryNum);

    assert(dataEntryNum >= 1);

    std::ifstream dataFile(dataFilePath, std::ios::binary);
    assert(dataFile);

    dataFile.seekg(static_cast<i64>((dataEntryNum - 1) * sizeof(StarwayDataEntry)), std::ios::beg);

    StarwayDataEntry entry;

    dataFile.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    assert(dataFile);

    entry.validate();

    std::array<char, 64> board;
    std::fill(board.begin(), board.end(), '.');

    // Fill board with the pieces
    while (entry.mOccupied > 0) {
        const Square sq = popLsb(entry.mOccupied);
        const Color pieceColor = static_cast<Color>(entry.mPieces & 0b1);
        const u8 pieceType = (entry.mPieces & 0b1110) >> 1;
        assert(pieceType <= static_cast<u8>(PieceType::King));

        constexpr std::array<char, 6> CHARS = {'P', 'N', 'B', 'R', 'Q', 'K'};

        board[static_cast<size_t>(sq)] = pieceColor == Color::White
                                             ? CHARS[pieceType]
                                             : static_cast<char>(std::tolower(CHARS[pieceType]));

        entry.mPieces >>= 4;
    }

    std::println("");

    // Print board
    for (i32 row = 7; row >= 0; row--) {
        for (i32 col = 0; col < 8; col++) {
            const size_t sq = static_cast<size_t>(row * 8 + col);
            std::print("{}{}", board[sq], col >= 7 ? "\n" : " ");
        }
    }

    std::println("");

    const u8 stmResult = static_cast<u8>(entry.get(Mask::STM_RESULT));
    const MontyformatMove bestMove = MontyformatMove(entry.mBestMove);

    std::println("In check: {}", bool(entry.get(Mask::IN_CHECK)));

    std::println("Stm score: {}", entry.mStmScore);

    std::println("Stm game result: {}",
                 stmResult == 0   ? "Lost"
                 : stmResult == 1 ? "Draw"
                 : stmResult == 2 ? "Won"
                                  : "Invalid");

    std::println("Best move: {}", bestMove.uci());
}
