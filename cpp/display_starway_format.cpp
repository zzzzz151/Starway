/*
Usage:
./display_starway_format
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
    std::println("Data file in Starway format: {}", dataFilePath);
    std::println("Data entry number from 1: {}", dataEntryNum);

    assert(dataEntryNum >= 1);

    std::ifstream dataFile(dataFilePath, std::ios::binary);
    assert(dataFile);

    size_t numRead = 0;
    StarwayDataEntry entry;

    // Read from input data file to StarwayDataEntry object until desired entry
    while (numRead < dataEntryNum) {
        entry = StarwayDataEntry(dataFile);
        numRead++;
    }

    entry.validate();

    const double stmScoreSigmoided =
        static_cast<double>(entry.mStmScore) / static_cast<double>(std::numeric_limits<u16>::max());

    const double stmResult = static_cast<double>(entry.get(Mask::STM_RESULT)) / 2.0f;

    std::println("Num pieces: {}", std::popcount(entry.mOccupied));
    std::println("Side to move: {}", entry.get(Mask::STM) == 0 ? "White" : "Black");
    std::println("In check: {}", entry.get(Mask::IN_CHECK) ? "true" : "false");
    std::println("Legal moves: {}", entry.get(Mask::NUM_MOVES));
    std::println("Stm score sigmoided: {:.4f}", stmScoreSigmoided);
    std::println("Stm game result (0.0, 0.5, 1.0): {}", stmResult);
}
