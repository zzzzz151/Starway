#include <fstream>
#include <iostream>
#include <limits>

#include "chess/types.hpp"
#include "chess/util.hpp"
#include "converter/data_entry.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ";
        std::cerr << argv[0];
        std::cerr << " <starway format file>";
        std::cerr << " <data entry number from 1>";
        std::cerr << std::endl;
        return 1;
    }

    // Read program args
    const std::string dataFileName = argv[1];
    const size_t dataEntryNum = std::stoull(argv[2]);

    // Print program args
    std::cout << "Starway data file: " << dataFileName << std::endl;
    std::cout << "Data entry number: " << dataEntryNum << std::endl;

    assert(dataEntryNum >= 1);

    std::ifstream dataFile(dataFileName, std::ios::binary);
    assert(dataFile);

    size_t numRead = 0;
    StarwayDataEntry entry;

    // Read from input data file to StarwayDataEntry object until desired entry
    while (numRead < dataEntryNum) {
        entry = StarwayDataEntry(dataFile);
        numRead++;
    }

    entry.validate();

    const Color stm = static_cast<Color>(entry.get(Mask::STM));
    const bool inCheck = entry.get(Mask::IN_CHECK);

    const double stmScoreSigmoided =
        static_cast<double>(entry.mStmScore) / static_cast<double>(std::numeric_limits<u16>::max());

    const float stmWdl = static_cast<float>(entry.get(Mask::WDL)) / 2.0f;

    std::cout << "Num pieces: " << std::popcount(entry.mOccupied) << std::endl;
    std::cout << "Side to move: " << (stm == Color::White ? "White" : "Black") << std::endl;
    std::cout << "In check: " << std::boolalpha << inCheck << std::noboolalpha << std::endl;
    std::cout << "Legal moves: " << entry.get(Mask::NUM_MOVES) << std::endl;
    std::cout << "Stm score sigmoided: " << stmScoreSigmoided << std::endl;
    std::cout << "Stm WDL: " << stmWdl << std::endl;
}
