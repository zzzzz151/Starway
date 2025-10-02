#pragma once

#include <print>

#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../utils.hpp"

constexpr u16 MIN_FULLMOVE_COUNTER = 9;
constexpr u8 MAX_HALFMOVE_CLOCK = 89;
constexpr i16 MAX_SCORE = 1838;
constexpr size_t MAX_LEGAL_MOVES_FILTER = 64;

struct DataFilter {
   private:
    size_t mInsufficientMaterial = 0;
    size_t mBadFullmoveCounter = 0;
    size_t mBadHalfmoveClock = 0;
    size_t mExtremeScore = 0;
    size_t mZeroLegalMoves = 0;
    size_t mTooManyMoves = 0;

   public:
    constexpr bool shouldSkip(const Position& pos, const i16 score, const size_t numMoves) {
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

        if (std::abs(score) > MAX_SCORE) {
            mExtremeScore++;
            skip = true;
        }

        if (numMoves == 0) {
            mZeroLegalMoves++;
            skip = true;
        }

        if (numMoves > MAX_LEGAL_MOVES_FILTER) {
            mTooManyMoves++;
            skip = true;
        }

        return skip;
    }

    constexpr void printStats() const {
        std::println("Filter counts:");
        std::println("  Insufficient material: {}", mInsufficientMaterial);
        std::println("  Fullmove counter < {}: {}", MIN_FULLMOVE_COUNTER, mBadFullmoveCounter);
        std::println("  Halfmove clock > {}: {}", MAX_HALFMOVE_CLOCK, mBadHalfmoveClock);
        std::println("  Abs(score) > {}: {}", MAX_SCORE, mExtremeScore);
        std::println("  No legal moves: {}", mZeroLegalMoves);
        std::println("  Legal moves > {}: {}", MAX_LEGAL_MOVES_FILTER, mTooManyMoves);
    }
};
