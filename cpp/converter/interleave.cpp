/*
Usage:
./interleave
    <input Starway data file>
    <output Starway data file>
    <converter's buffer capacity>
    <batch offsets input file>
    <batch offsets output file>
    <batch size>
*/

#include <cassert>
#include <cmath>
#include <fstream>
#include <random>

#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../utils.hpp"
#include "data_entry.hpp"

// Each chunk is a dump of the montyformat_to_starway buffer (N batches)
struct ShuffledChunkOfBatches {
   public:
    std::ifstream ifstream;
    size_t numDataEntries;
};

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: ";
        std::cerr << argv[0];
        std::cerr << " <input Starway data file>";
        std::cerr << " <output Starway data file>";
        std::cerr << " <converter's buffer capacity>";
        std::cerr << " <batch offsets input file>";
        std::cerr << " <batch offsets output file>";
        std::cerr << " <batch size>";
        std::cerr << std::endl;
        return 1;
    }

    // Read program args
    const std::string inputDataFileName = argv[1];
    const std::string outDataFileName = argv[2];
    const size_t bufferCapacity = std::stoull(argv[3]);
    const std::string batchOffsetsInputFileName = argv[4];
    const std::string batchOffsetsOutFileName = argv[5];
    const size_t batchSize = std::stoull(argv[6]);

    // Print program args
    std::cout << "Input data file: " << inputDataFileName << std::endl;
    std::cout << "Output data file: " << outDataFileName << std::endl;
    std::cout << "Converter's buffer capacity: " << bufferCapacity << " data entries" << std::endl;
    std::cout << "Batch offsets input file: " << batchOffsetsInputFileName << std::endl;
    std::cout << "Batch offsets output file: " << batchOffsetsOutFileName << std::endl;
    std::cout << "Batch size: " << batchSize << " data entries" << std::endl;

    assert(bufferCapacity > 0);
    assert(batchSize > 0);
    assert(bufferCapacity % batchSize == 0);

    // Open files
    std::ifstream inputDataFile(inputDataFileName, std::ios::binary);
    std::ofstream outDataFile(outDataFileName, std::ios::binary);
    std::ifstream batchOffsetsInputFile(batchOffsetsInputFileName, std::ios::binary);
    std::ofstream batchOffsetsOutFile(batchOffsetsOutFileName, std::ios::binary);

    assert(inputDataFile);
    assert(outDataFile);
    assert(batchOffsetsInputFile);
    assert(batchOffsetsOutFile);

    // Prepare batch offsets vector
    std::vector<size_t> batchOffsets;
    batchOffsetsInputFile.seekg(0, std::ios::end);
    batchOffsets.resize(static_cast<size_t>(batchOffsetsInputFile.tellg()) / sizeof(size_t));
    batchOffsetsInputFile.seekg(0, std::ios::beg);

    // Move batch offsets from batch offsets input file into RAM
    batchOffsetsInputFile.read(reinterpret_cast<char*>(batchOffsets.data()),
                               static_cast<i64>(batchOffsets.size() * sizeof(size_t)));

    assert(batchOffsetsInputFile);

    std::cout << "Batches: " << batchOffsets.size() << "\n" << std::endl;
    assert(batchOffsets.size() > 0);

    std::vector<ShuffledChunkOfBatches> chunks;

    chunks.resize([&]() {
        const double x = static_cast<double>(batchOffsets.size() * batchSize) /
                         static_cast<double>(bufferCapacity);

        return static_cast<size_t>(std::ceil(x));
    }());

    // Initialize the chunks
    // Seek the ifstream of each chunk to where it starts in the input data file
    for (size_t i = 0; i < chunks.size(); i++) {
        ShuffledChunkOfBatches& chunk = chunks[i];

        chunk.ifstream = std::ifstream(inputDataFileName, std::ios::binary);
        assert(chunk.ifstream);

        const size_t offset = batchOffsets[bufferCapacity / batchSize * i];

        chunk.ifstream.seekg(static_cast<i64>(offset), std::ios::beg);
        assert(chunk.ifstream);

        chunk.numDataEntries = &chunk != &chunks.back()
                                   ? bufferCapacity
                                   : (batchOffsets.size() * batchSize) % bufferCapacity;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());

    size_t dataEntriesLeft = batchOffsets.size() * batchSize;

    while (dataEntriesLeft > 0) {
        size_t entryIdx = gen() % dataEntriesLeft;
        size_t chunkIdx = 0;

        while (chunks[chunkIdx].numDataEntries < entryIdx) {
            assert(entryIdx >= chunks[chunkIdx].numDataEntries);
            entryIdx -= chunks[chunkIdx].numDataEntries;
            chunkIdx++;
            assert(chunkIdx < chunks.size());
        }

        ShuffledChunkOfBatches& chunk = chunks[chunkIdx];

        // Read from this ifstream of the input data file to StarwayDataEntry object
        StarwayDataEntry entry;
        chunk.ifstream.read(reinterpret_cast<char*>(&entry.miscData), sizeof(entry.miscData));
        chunk.ifstream.read(reinterpret_cast<char*>(&entry.occupied), sizeof(entry.occupied));
        chunk.ifstream.read(reinterpret_cast<char*>(&entry.pieces), sizeof(entry.pieces));
        chunk.ifstream.read(reinterpret_cast<char*>(&entry.stmScore), sizeof(entry.stmScore));

        // Some data entry validation
        assert(entry.get(Mask::EP_FILE) <= 8);
        assert(entry.get(Mask::WDL) <= 2);
        const size_t numPieces = static_cast<size_t>(std::popcount(entry.occupied));
        const size_t numMoves = entry.get(Mask::NUM_MOVES);
        assert(numPieces > 2 && numPieces <= 32);
        assert(numMoves > 0 && numMoves <= entry.visits.size());

        // For the visits array, we only read the filled elements (number of legal moves)
        chunk.ifstream.read(reinterpret_cast<char*>(&entry.visits), entry.visitsBytesCount());

        assert(chunk.ifstream);

        // If starting to write a new batch, save the batch offset
        if (dataEntriesLeft % batchSize == 0) {
            const size_t batchesWritten =
                (batchOffsets.size() * batchSize - dataEntriesLeft) / batchSize;

            batchOffsets[batchesWritten] = static_cast<size_t>(outDataFile.tellp());
        }

        // Write data entry to output data file
        outDataFile.write(reinterpret_cast<const char*>(&entry.miscData), sizeof(entry.miscData));
        outDataFile.write(reinterpret_cast<const char*>(&entry.occupied), sizeof(entry.occupied));
        outDataFile.write(reinterpret_cast<const char*>(&entry.pieces), sizeof(entry.pieces));
        outDataFile.write(reinterpret_cast<const char*>(&entry.stmScore), sizeof(entry.stmScore));

        // For the visits array, we only write the filled elements (number of legal moves)
        outDataFile.write(reinterpret_cast<const char*>(&entry.visits), entry.visitsBytesCount());

        chunk.numDataEntries--;
        dataEntriesLeft--;

        // If all data entries of this chunk have been written, remove it from the chunks vector
        if (chunk.numDataEntries <= 0) {
            chunks.erase(chunks.begin() + static_cast<i64>(chunkIdx));
        }

        // Log progress once in a while
        if (dataEntriesLeft % 16'777'216 == 0) {
            std::cout << "Data entries written: ";
            std::cout << (batchOffsets.size() * batchSize - dataEntriesLeft);
            std::cout << std::endl;
        }
    }

    // Write the new batch offsets to batch offsets output file
    batchOffsetsOutFile.write(reinterpret_cast<const char*>(batchOffsets.data()),
                              static_cast<i64>(batchOffsets.size() * sizeof(size_t)));

    std::cout << "\nFinished";
    std::cout << "Data entries written: ";
    std::cout << (batchOffsets.size() * batchSize);
    std::cout << std::endl;

    // Assert input and output data files have same size
    inputDataFile.seekg(0, std::ios::end);
    assert(static_cast<size_t>(inputDataFile.tellg()) == static_cast<size_t>(outDataFile.tellp()));

    return 0;
}
