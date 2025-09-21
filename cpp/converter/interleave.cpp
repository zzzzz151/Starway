/*
Usage:
./interleave
    <input data file in Starway format>
    <output data file in Starway format>
    <converter's buffer capacity>
    <batch offsets input file>
    <batch offsets output file>
    <batch size>
*/

#include <cassert>
#include <cmath>
#include <fstream>
#include <print>
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
        std::println(std::cerr,
                     "Usage: {} {} {} {} {} {} {}",
                     argv[0],
                     "<input data file in Starway format>",
                     "<output data file in Starway format>",
                     "<converter's buffer capacity>",
                     "<batch offsets input file>",
                     "<batch offsets output file>",
                     "<batch size>");

        return 1;
    }

    // Read program args
    const std::string inputDataFilePath = argv[1];
    const std::string outDataFilePath = argv[2];
    const size_t bufferCapacity = std::stoull(argv[3]);
    const std::string batchOffsetsInputFilePath = argv[4];
    const std::string batchOffsetsOutFilePath = argv[5];
    const size_t batchSize = std::stoull(argv[6]);

    // Print program args
    std::println("Input data file: {}", inputDataFilePath);
    std::println("Output data file: {}", outDataFilePath);
    std::println("Converter's buffer capacity: {} data entries", bufferCapacity);
    std::println("Batch offsets input file: {}", batchOffsetsInputFilePath);
    std::println("Batch offsets output file: {}", batchOffsetsOutFilePath);
    std::println("Batch size: {} data entries", batchSize);

    assert(bufferCapacity > 0);
    assert(batchSize > 0);
    assert(bufferCapacity % batchSize == 0);

    // Open files
    std::ifstream inputDataFile(inputDataFilePath, std::ios::binary);
    std::ofstream outDataFile(outDataFilePath, std::ios::binary);
    std::ifstream batchOffsetsInputFile(batchOffsetsInputFilePath, std::ios::binary);
    std::ofstream batchOffsetsOutFile(batchOffsetsOutFilePath, std::ios::binary);

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

    std::println("Batches: {}", batchOffsets.size());
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

        chunk.ifstream = std::ifstream(inputDataFilePath, std::ios::binary);
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
        StarwayDataEntry entry = StarwayDataEntry(chunk.ifstream);
        entry.validate();

        // If starting to write a new batch, save the batch offset
        if (dataEntriesLeft % batchSize == 0) {
            const size_t batchesWritten =
                (batchOffsets.size() * batchSize - dataEntriesLeft) / batchSize;

            batchOffsets[batchesWritten] = static_cast<size_t>(outDataFile.tellp());
        }

        // Write data entry to output data file
        entry.writeToOut(outDataFile);

        chunk.numDataEntries--;
        dataEntriesLeft--;

        // If all data entries of this chunk have been written, remove it from the chunks vector
        if (chunk.numDataEntries <= 0) {
            chunks.erase(chunks.begin() + static_cast<i64>(chunkIdx));
        }

        // Log progress once in a while
        if (dataEntriesLeft % 16'777'216 == 0) {
            std::println("Data entries written: {}",
                         batchOffsets.size() * batchSize - dataEntriesLeft);
        }
    }

    // Write the new batch offsets to batch offsets output file
    batchOffsetsOutFile.write(reinterpret_cast<const char*>(batchOffsets.data()),
                              static_cast<i64>(batchOffsets.size() * sizeof(size_t)));

    std::println("Finished; wrote {} data entries",
                 batchOffsets.size() * batchSize - dataEntriesLeft);

    // Assert input and output data files have same size
    inputDataFile.seekg(0, std::ios::end);
    assert(static_cast<size_t>(inputDataFile.tellg()) == static_cast<size_t>(outDataFile.tellp()));

    return 0;
}
