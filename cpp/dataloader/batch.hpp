#pragma once

#include <span>

#include "../utils.hpp"

constexpr size_t MAX_PIECES_PER_POS = 32;
constexpr size_t POLICY_OUTPUT_SIZE = 1882;

// A batch of N data entries (1 data entry = 1 position)
struct Batch {
   public:
    // [entryIdx][MAX_PIECES_PER_POS] arrays
    // Dataloader will initialize all elements to -1 (no piece) then fill these arrays
    i16* activeFeaturesStm;
    i16* activeFeaturesNtm;

    // [entryIdx] arrays
    i16* stmScores;
    float* stmWDLs;

    // [entryIdx][moveIdx] array, where moveIdx comes from move_mapping.hpp
    // Illegal moves are set to a large negative number
    // Legal logits are set to the visit distribution of that data entry
    i16* logits;

    constexpr Batch(const std::size_t batchSize) {
        this->activeFeaturesStm = new i16[batchSize * MAX_PIECES_PER_POS];
        this->activeFeaturesNtm = new i16[batchSize * MAX_PIECES_PER_POS];

        this->stmScores = new i16[batchSize];
        this->stmWDLs = new float[batchSize];

        this->logits = new i16[batchSize * POLICY_OUTPUT_SIZE];
    }

};  // struct Batch
