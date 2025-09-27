#pragma once

#include <span>

#include "../utils.hpp"

constexpr size_t MAX_PIECES_PER_POS = 32;

// Should match the MAX_LEGAL_MOVES_FILTER in converter/data_filter.hpp
// Do not rename this constant since the python code grabs it
constexpr size_t MAX_MOVES_PER_POS = 64;

// A batch of N data entries (1 data entry = 1 position)
struct Batch {
   public:
    // [entryIdx][MAX_PIECES_PER_POS] arrays
    // Dataloader will initialize all elements to -1 (no piece) then fill these arrays
    i16* activeFeaturesStm;
    i16* activeFeaturesNtm;

    // [entryIdx] arrays
    float* stmScoresSigmoided;
    float* stmResults;

    size_t totalLegalMoves;
    float* legalMovesIdxsAndVisitsPercent;  // Stores tuples (entryIdx, moveIdx, visitsPercent)

    constexpr Batch(const std::size_t batchSize) {
        this->activeFeaturesStm = new i16[batchSize * MAX_PIECES_PER_POS];
        this->activeFeaturesNtm = new i16[batchSize * MAX_PIECES_PER_POS];

        this->stmScoresSigmoided = new float[batchSize];
        this->stmResults = new float[batchSize];

        this->totalLegalMoves = 0;
        this->legalMovesIdxsAndVisitsPercent = new float[batchSize * MAX_MOVES_PER_POS * 3];
    }

};  // struct Batch
