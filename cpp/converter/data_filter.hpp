#pragma once

#include <limits>
#include <print>

#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../utils.hpp"

constexpr u16 MIN_FULLMOVE_COUNTER = 9;
constexpr u8 MAX_HALFMOVE_CLOCK = 89;
constexpr size_t MAX_LEGAL_MOVES_FILTER = 64;
constexpr double MIN_SCORE_SIGMOIDED = 0.01;
constexpr double MAX_SCORE_SIGMOIDED = 1.0 - MIN_SCORE_SIGMOIDED;

static_assert(MIN_SCORE_SIGMOIDED > 0.0 && MIN_SCORE_SIGMOIDED < 1.0);
static_assert(MAX_SCORE_SIGMOIDED > 0.0 && MAX_SCORE_SIGMOIDED < 1.0);
static_assert(MIN_SCORE_SIGMOIDED < MAX_SCORE_SIGMOIDED);

struct DataFilter {
   private:
    size_t mInsufficientMaterial = 0;
    size_t mBadFullmoveCounter = 0;
    size_t mBadHalfmoveClock = 0;
    size_t mTooManyMoves = 0;
    size_t mExtremeScore = 0;
    size_t mBestMoveZeroVisits = 0;

   public:
    constexpr bool shouldSkip(const Position& pos,
                              const size_t numMoves,
                              const double scoreSigmoided,
                              const u8 bestMoveVisits) {
        bool skip = false;

        if (pos.isInsufficientMaterial()) {
            mInsufficientMaterial++;
            skip = true;
        }

        if (pos.getFullMoveCounter() < MIN_FULLMOVE_COUNTER) {
            mBadFullmoveCounter++;
            skip = true;
        }

        if (pos.getHalfMoveClock() > MAX_HALFMOVE_CLOCK) {
            mBadHalfmoveClock++;
            skip = true;
        }

        if (numMoves > MAX_LEGAL_MOVES_FILTER) {
            mTooManyMoves++;
            skip = true;
        }

        if (scoreSigmoided < MIN_SCORE_SIGMOIDED || scoreSigmoided > MAX_SCORE_SIGMOIDED) {
            mExtremeScore++;
            skip = true;
        }

        if (bestMoveVisits == 0) {
            mBestMoveZeroVisits++;
            skip = true;
        }

        return skip;
    }

    constexpr void printCounts() const {
        std::println("Filter counts:");
        std::println("  Insufficient material: {}", mInsufficientMaterial);
        std::println("  Fullmove counter < {}: {}", MIN_FULLMOVE_COUNTER, mBadFullmoveCounter);
        std::println("  Halfmove clock > {}: {}", MAX_HALFMOVE_CLOCK, mBadHalfmoveClock);
        std::println("  Legal moves > {}: {}", MAX_LEGAL_MOVES_FILTER, mTooManyMoves);

        std::println("  Score sigmoided < {} or > {}: {}",
                     MIN_SCORE_SIGMOIDED,
                     MAX_SCORE_SIGMOIDED,
                     mExtremeScore);

        std::println("  Best move has 0 visits: {}", mBestMoveZeroVisits);
    }
};
