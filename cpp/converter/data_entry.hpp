#pragma once

#include <bit>
#include <cassert>
#include <fstream>

#include "../chess/position.hpp"
#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../utils.hpp"

// Masks for StarwayDataEntry.mMiscData
// "x-y" includes both x-th and y-th bits
enum class Mask : u32 {
    // 1st lowest bit: set if black to move
    STM = 1u,

    // 2nd lowest bit: set if in check
    IN_CHECK = 1u << 1,

    // 3-8: side-to-move (always white when oriented) king square oriented
    OUR_KING_SQ_ORIENTED = 0b111'111u << 2,

    // 9-14: not-side-to-move (always black when oriented) king square oriented
    THEIR_KING_SQ_ORIENTED = 0b111'111u << 8,

    // 15: set if stm (always white since it's oriented) short castling right
    CASTLING_KS = 1u << 14,

    // 16: set if stm (always white since it's oriented) long castling right
    CASTLING_QS = 1u << 15,

    // 17-20: en passant file (8 if none)
    EP_FILE = 0b1111u << 16,

    // 21-22: Game result (0 if stm lost, 1 if draw, 2 if stm won)
    STM_RESULT = 0b11u << 20,

    // 23-32: unused
};

struct StarwayDataEntry {
   public:
    u32 mMiscData;  // See Mask enum for the encoding

    u64 mOccupied;  // Oriented (flipped vertically if black to move)

    // 4 bits per oriented piece for a max of 32 oriented pieces
    // Lsb of the 4 bits is set if the color of the oriented piece is black
    // Other 3 bits is piece type (0-5 including both)
    u128 mPieces;

    i16 mStmScore;

    u16 mBestMove;  // Oriented (flipped vertically if black to move)

    constexpr StarwayDataEntry() {}  // Does not init fields

    // Get some field from misc data
    constexpr u32 get(const Mask mask) const {
        const u32 maskU32 = static_cast<u32>(mask);
        return (mMiscData & maskU32) >> std::countr_zero(maskU32);
    }

    // Set some field in misc data
    constexpr void set(const Mask mask, const u32 value) {
        const u32 maskU32 = static_cast<u32>(mask);
        assert(value <= (maskU32 >> std::countr_zero(maskU32)));

        mMiscData &= ~maskU32;
        mMiscData |= value << std::countr_zero(maskU32);
    }

    // Calculate and set mMiscData
    constexpr void setMiscData(const Position& pos, const u8 stmResult) {
        mMiscData = 0;

        assert(stmResult <= 2);

        const Square ourKingSqOriented =
            maybeRankFlipped(pos.getKingSq(pos.mSideToMove), pos.mSideToMove);

        const Square theirKingSqOriented =
            maybeRankFlipped(pos.getKingSq(!pos.mSideToMove), pos.mSideToMove);

        set(Mask::STM, static_cast<u32>(pos.mSideToMove));
        set(Mask::IN_CHECK, pos.getCheckers() > 0);
        set(Mask::OUR_KING_SQ_ORIENTED, static_cast<u32>(ourKingSqOriented));
        set(Mask::THEIR_KING_SQ_ORIENTED, static_cast<u32>(theirKingSqOriented));
        set(Mask::CASTLING_KS, pos.hasCastlingRight(pos.mSideToMove, true));
        set(Mask::CASTLING_QS, pos.hasCastlingRight(pos.mSideToMove, false));

        if (pos.getEpSquare().has_value()) {
            const Square epSquare = *(pos.getEpSquare());
            set(Mask::EP_FILE, static_cast<u32>(fileOf(epSquare)));
        } else {
            // Store file 8 if no ep square
            set(Mask::EP_FILE, 8);
        }

        set(Mask::STM_RESULT, stmResult);
    }

    // Calculate and set mOccupied and mPieces
    constexpr void setOccAndPieces(const Position& pos) {
        mOccupied = 0;
        mPieces = 0;

        // Occupancy, flipped vertically if black to move
        u64 occOriented =
            pos.mSideToMove == Color::White ? pos.getOcc() : __builtin_bswap64(pos.getOcc());

        while (occOriented > 0) {
            const Square sq = popLsb(occOriented);

            auto [pieceColor, pieceType] =
                pos.pieceAt(maybeRankFlipped(sq, pos.mSideToMove)).value();

            if (pos.mSideToMove == Color::Black) {
                pieceColor = !pieceColor;
            }

            if (pieceType == PieceType::King) {
                const Mask mask = pieceColor == Color::White ? Mask::OUR_KING_SQ_ORIENTED
                                                             : Mask::THEIR_KING_SQ_ORIENTED;

                assert(sq == static_cast<Square>(get(mask)));
            }

            const u128 fourBitsPiece =
                static_cast<u128>(pieceColor) | (static_cast<u128>(pieceType) << 1);

            mPieces |= fourBitsPiece << (std::popcount(mOccupied) * 4);
            mOccupied |= sqToBb(sq);
        }
    }

    constexpr void validate() const {
        assert(get(Mask::EP_FILE) <= 8);
        assert(get(Mask::STM_RESULT) <= 2);
        assert(std::popcount(mOccupied) > 2 && std::popcount(mOccupied) <= 32);
        assert(bbContainsSq(mOccupied, static_cast<Square>(get(Mask::OUR_KING_SQ_ORIENTED))));
        assert(bbContainsSq(mOccupied, static_cast<Square>(get(Mask::THEIR_KING_SQ_ORIENTED))));
        assert(mBestMove > 0);
    }

} __attribute__((packed));  // struct StarwayDataEntry

static_assert(sizeof(StarwayDataEntry) == 32);  // 32 bytes
