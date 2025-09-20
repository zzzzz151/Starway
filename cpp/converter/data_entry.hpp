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

    // 21-22: WDL (0 if stm lost, 1 if draw, 2 if stm won)
    WDL = 0b11u << 20,

    // 23-30: number of legal moves
    NUM_MOVES = 0b1111'1111u << 22

    // 31-32: unused
};

// How the visits are encoded:
// https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#visit-distribution
struct MoveAndVisits {
   public:
    u16 move;
    u8 visits;
} __attribute__((packed));

struct StarwayDataEntry {
   public:
    u32 mMiscData;  // See Mask enum for the encoding

    u64 mOccupied;  // Oriented (flipped vertically if black to move)

    // 4 bits per oriented piece for a max of 32 oriented mPieces
    // Lsb of the 4 bits is set if the color of the oriented piece is black
    // Other 3 bits is piece type (0-5 including both)
    u128 mPieces;

    u16 mStmScore;  // Divide by u16 max to get stm score sigmoided

    // The number of filled MoveAndVisits elements is the number of legal moves
    // The u16 move is oriented (flipped vertically if black to move)
    std::array<MoveAndVisits, 218> mVisits;

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
    constexpr void setMiscData(const Position& pos, const u8 stmWdl, const u8 numMoves) {
        mMiscData = 0;

        assert(stmWdl <= 2);
        assert(numMoves > 0 && static_cast<size_t>(numMoves) <= mVisits.size());

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

        set(Mask::WDL, stmWdl);
        set(Mask::NUM_MOVES, numMoves);
    }

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

    // How many bytes should be occupied by the filled elements of the mVisits member field?
    constexpr size_t visitsBytesCount() const {
        const size_t numMoves = get(Mask::NUM_MOVES);
        assert(numMoves > 0 && numMoves <= mVisits.size());

        return numMoves * sizeof(MoveAndVisits);
    }

    constexpr void validate() const {
        assert(get(Mask::EP_FILE) <= 8);
        assert(get(Mask::WDL) <= 2);
        assert(get(Mask::NUM_MOVES) > 0 && get(Mask::NUM_MOVES) <= mVisits.size());
        assert(std::popcount(mOccupied) > 2 && std::popcount(mOccupied) <= 32);
        assert(bbContainsSq(mOccupied, static_cast<Square>(get(Mask::OUR_KING_SQ_ORIENTED))));
        assert(bbContainsSq(mOccupied, static_cast<Square>(get(Mask::THEIR_KING_SQ_ORIENTED))));
    }

} __attribute__((packed));  // struct StarwayDataEntry
