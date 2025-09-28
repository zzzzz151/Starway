#pragma once

#include <span>

#include "../utils.hpp"

constexpr size_t MAX_PIECES_PER_POS = 32;

// Should match the MAX_LEGAL_MOVES_FILTER in converter/data_filter.hpp
// Must match the MAX_MOVES_PER_POS in python/settings.py
constexpr size_t MAX_MOVES_PER_POS = 64;

// A batch of N data entries (1 data entry = 1 position)
struct Batch {
   public:
    // [entryIdx][MAX_PIECES_PER_POS] arrays padded with -1
    i16* activeFeaturesStm;
    i16* activeFeaturesNtm;

    // [entryIdx] arrays
    float* stmScoresSigmoided;
    float* stmResults;

    // [entryIdx][MAX_MOVES_PER_POS] array padded with -1
    i16* legalMovesIdxs;

    // [entryIdx][MAX_MOVES_PER_POS] array padded with 0.0
    float* visitsPercent;

    constexpr Batch(const std::size_t batchSize) {
        this->activeFeaturesStm = new i16[batchSize * MAX_PIECES_PER_POS];
        this->activeFeaturesNtm = new i16[batchSize * MAX_PIECES_PER_POS];

        this->stmScoresSigmoided = new float[batchSize];
        this->stmResults = new float[batchSize];

        this->legalMovesIdxs = new i16[batchSize * MAX_MOVES_PER_POS];

        this->visitsPercent = new float[batchSize * MAX_MOVES_PER_POS];
    }

};  // struct Batch
