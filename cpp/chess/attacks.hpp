#pragma once

// incbin fuckery
#ifdef _MSC_VER
#define STARWAY_MSVC
#pragma push_macro("_MSC_VER")
#undef _MSC_VER
#endif

#include <cassert>

#include "../incbin.h"
#include "../utils.hpp"
#include "types.hpp"
#include "util.hpp"

INCBIN(PawnAttacks, "cpp/chess/embeds/pawn_attacks.bin");
INCBIN(KnightAttacks, "cpp/chess/embeds/knight_attacks.bin");
INCBIN(BishopAttacks, "cpp/chess/embeds/bishop_attacks.bin");
INCBIN(RookAttacks, "cpp/chess/embeds/rook_attacks.bin");
INCBIN(KingAttacks, "cpp/chess/embeds/king_attacks.bin");
INCBIN(BetweenExclusiveBbs, "cpp/chess/embeds/between_exclusive.bin");
INCBIN(LineThruBbs, "cpp/chess/embeds/line_thru.bin");

template <std::size_t ATTACKS_TABLE_SIZE>
struct MagicEntry {
   private:
    u64 mAtksEmptyBoardExcludingLastSqEachDir;
    u64 mMagic;
    u64 mShift;
    std::array<u64, ATTACKS_TABLE_SIZE> mAtksByKey;

   public:
    constexpr u64 attacks(const u64 occ) const {
        const u64 blockers = occ & mAtksEmptyBoardExcludingLastSqEachDir;
        const std::size_t idx = (blockers * mMagic) >> mShift;
        assert(idx < ATTACKS_TABLE_SIZE);
        return mAtksByKey[idx];
    }
};

const auto PAWN_ATTACKS = *reinterpret_cast<const MultiArray<u64, 2, 64>*>(gPawnAttacksData);

const auto KNIGHT_ATTACKS = *reinterpret_cast<const std::array<u64, 64>*>(gKnightAttacksData);

const auto BISHOP_ATTACKS =
    *reinterpret_cast<const std::array<MagicEntry<512>, 64>*>(gBishopAttacksData);

const auto ROOK_ATTACKS =
    *reinterpret_cast<const std::array<MagicEntry<4096>, 64>*>(gRookAttacksData);

const auto KING_ATTACKS = *reinterpret_cast<const std::array<u64, 64>*>(gKingAttacksData);

const auto BETWEEN_EXCLUSIVE_BB =
    *reinterpret_cast<const MultiArray<u64, 64, 64>*>(gBetweenExclusiveBbsData);

const auto LINE_THRU_BB = *reinterpret_cast<const MultiArray<u64, 64, 64>*>(gLineThruBbsData);

constexpr u64 getQueenAttacks(const Square sq, const u64 occ) {
    return BISHOP_ATTACKS[static_cast<std::size_t>(sq)].attacks(occ) |
           ROOK_ATTACKS[static_cast<std::size_t>(sq)].attacks(occ);
}
