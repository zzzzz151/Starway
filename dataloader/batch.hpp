// clang-format off

#pragma once

#include "utils.hpp"

struct Batch {
public:

    std::size_t numActiveFeatures = 0;

    i16* activeFeaturesStm;
    i16* activeFeaturesNtm;

    i16* stmScores;
    float* stmWDLs;

    i16* bestMoveIdx1882;

    std::size_t totalLegalMoves = 0;
    i16* legalMovesIdxs1882;

    constexpr Batch(const std::size_t batchSize) {
        // Array size is * 2 because the features are (positionIndex, feature)
        // AKA a (numActiveFeatures, 2) matrix
        activeFeaturesStm = new i16[batchSize * 32];
        activeFeaturesNtm = new i16[batchSize * 32];

        stmScores = new i16[batchSize];
        stmWDLs   = new float[batchSize];

        bestMoveIdx1882 = new i16[batchSize];

        // Train data only has positions with more than 0 legal moves and less than 128
        legalMovesIdxs1882 = new i16[batchSize * 127 * 2];
    }

}; // struct Batch
