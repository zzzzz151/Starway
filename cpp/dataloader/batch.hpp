#pragma once

#include "../utils.hpp"

constexpr size_t MAX_LEGAL_MOVES = 218;

struct Batch {
   public:
    i16* activeFeaturesStm;
    i16* activeFeaturesNtm;

    i16* stmScores;
    float* stmWDLs;

    std::size_t totalLegalMoves;
    i16* legalMovesIdx;
    u8* visits;

    constexpr Batch(const std::size_t batchSize) {
        this->activeFeaturesStm = new i16[batchSize * 32];
        this->activeFeaturesNtm = new i16[batchSize * 32];

        this->stmScores = new i16[batchSize];
        this->stmWDLs = new float[batchSize];

        this->totalLegalMoves = 0;
        this->legalMovesIdx = new i16[batchSize * MAX_LEGAL_MOVES];
        this->visits = new u8[batchSize * MAX_LEGAL_MOVES];
    }

};  // struct Batch
