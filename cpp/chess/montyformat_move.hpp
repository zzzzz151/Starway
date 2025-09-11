#pragma once

// incbin fuckery
#ifdef _MSC_VER
#define STARWAY_MSVC
#pragma push_macro("_MSC_VER")
#undef _MSC_VER
#endif

#include <array>
#include <cassert>
#include <optional>
#include <string>

#include "../incbin.h"
#include "../utils.hpp"
#include "types.hpp"
#include "util.hpp"

INCBIN(MovesMap1880, "moves_map_1880.bin");

const MultiArray<i16, 64, 64, 7> MOVES_MAP_1880 =
    *reinterpret_cast<const MultiArray<i16, 64, 64, 7>*>(gMovesMap1880Data);

enum class MfMoveFlag : u16 {
    Quiet = 0,
    PawnDoublePush = 1,
    CastlingKs = 2,
    CastlingQs = 3,
    Capture = 4,
    EnPassant = 5,
    KnightPromo = 8,
    BishopPromo = 9,
    RookPromo = 10,
    QueenPromo = 11,
    KnightPromoCapture = 12,
    BishopPromoCapture = 13,
    RookPromoCapture = 14,
    QueenPromoCapture = 15
};

struct MontyformatMove {
   private:
    u16 mMove;

   public:
    constexpr MontyformatMove() {}  // Does not init mMove

    constexpr MontyformatMove(const u16 mfMove) { mMove = mfMove; }

    constexpr MontyformatMove(const Square src, const Square dst, const MfMoveFlag flag) {
        mMove = static_cast<u16>(static_cast<u16>(src) << 10);
        mMove |= static_cast<u16>(dst) << 4;
        mMove |= static_cast<u16>(flag);
    }

    constexpr bool operator==(const MontyformatMove other) const { return mMove == other.mMove; }

    constexpr bool operator!=(const MontyformatMove other) const { return mMove != other.mMove; }

    constexpr u16 asU16() const { return mMove; }

    constexpr bool isNull() const { return mMove == 0; }

    constexpr Square getSrc() const {
        const u8 sq = mMove >> 10;
        assert(sq < 64);
        return static_cast<Square>(sq);
    }

    constexpr Square getDst() const {
        const u8 sq = (mMove >> 4) & 0b111'111;
        assert(sq < 64);
        return static_cast<Square>(sq);
    }

   private:
    constexpr MfMoveFlag getFlag() const {
        const u16 flag = mMove & 0b1111;
        assert(flag != 6 && flag != 7);
        return static_cast<MfMoveFlag>(flag);
    }

   public:
    constexpr bool isCapture() const {
        const MfMoveFlag flag = getFlag();

        return flag == MfMoveFlag::Capture || flag == MfMoveFlag::EnPassant ||
               static_cast<u16>(flag) >= static_cast<u16>(MfMoveFlag::KnightPromoCapture);
    }

    constexpr bool isKsCastling() const { return getFlag() == MfMoveFlag::CastlingKs; }

    constexpr bool isQsCastling() const { return getFlag() == MfMoveFlag::CastlingQs; }

    constexpr bool isPromo() const {
        return static_cast<u16>(getFlag()) >= static_cast<u16>(MfMoveFlag::KnightPromo);
    }

    constexpr std::optional<PieceType> getPromoPt() const {
        if (!isPromo()) {
            return std::nullopt;
        }

        const u32 flag = static_cast<u16>(getFlag());
        return static_cast<PieceType>(flag % 4 + 1);
    }

    constexpr bool isEnPassant() const { return getFlag() == MfMoveFlag::EnPassant; }

    constexpr bool isPawnDoublePush() const { return getFlag() == MfMoveFlag::PawnDoublePush; }

    constexpr std::string uci() const {
        const Square src = getSrc();
        const Square dst = getDst();
        const std::optional<PieceType> promoPt = getPromoPt();

        std::string res = squareToStr(src) + squareToStr(dst);

        if (promoPt.has_value()) {
            res += *promoPt == PieceType::Knight   ? 'n'
                   : *promoPt == PieceType::Bishop ? 'b'
                   : *promoPt == PieceType::Rook   ? 'r'
                                                   : 'q';
        }

        return res;
    }

    constexpr MontyformatMove maybeRanksFlipped(const Color stm) const {
        if (stm == Color::White) {
            return *this;
        }

        const Square newSrc = rankFlipped(getSrc());
        const Square newDst = rankFlipped(getDst());
        return MontyformatMove(newSrc, newDst, getFlag());
    }

    constexpr void validate(const bool whiteToMove) const {
        assert(!isNull());

        const Square src = getSrc();
        const Square dst = getDst();

        assert(src != dst);

        const MfMoveFlag flag = getFlag();

        if (flag == MfMoveFlag::PawnDoublePush) {
            assert(rankOf(src) == (whiteToMove ? Rank::Rank2 : Rank::Rank7));
            assert(rankOf(dst) == (whiteToMove ? Rank::Rank4 : Rank::Rank5));
        }

        if (isKsCastling() || isQsCastling()) {
            assert(src == (whiteToMove ? Square::E1 : Square::E8));
        }

        if (isKsCastling()) {
            assert(dst == (whiteToMove ? Square::G1 : Square::G8));
        }

        if (isQsCastling()) {
            assert(dst == (whiteToMove ? Square::C1 : Square::C8));
        }

        // Pawn can only promote in backrank
        if (isPromo()) {
            assert(rankOf(src) == (whiteToMove ? Rank::Rank7 : Rank::Rank2));
            assert(rankOf(dst) == (whiteToMove ? Rank::Rank8 : Rank::Rank1));
        }

        if (isEnPassant()) {
            assert(rankOf(src) == (whiteToMove ? Rank::Rank5 : Rank::Rank4));
            assert(rankOf(dst) == (whiteToMove ? Rank::Rank6 : Rank::Rank3));
        }

        const size_t firstIdx = static_cast<size_t>(whiteToMove ? src : rankFlipped(src));
        const size_t secondIdx = static_cast<size_t>(whiteToMove ? dst : rankFlipped(dst));
        const size_t thirdIdx = isPromo() ? static_cast<size_t>(*getPromoPt()) : 6;

        assert(MOVES_MAP_1880[firstIdx][secondIdx][thirdIdx] >= 0);
    }

};  // struct MontyformatMove
