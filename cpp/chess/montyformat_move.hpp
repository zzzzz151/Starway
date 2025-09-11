#pragma once

#include <array>
#include <cassert>
#include <optional>
#include <string>

#include "../utils.hpp"
#include "types.hpp"
#include "util.hpp"

// Montyformat move encoding:
// https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#moves-and-their-associated-information

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

    constexpr void validate(const bool whiteToMove, const PieceType pt) const {
        assert(!isNull());

        const Square src = getSrc();
        const Square dst = getDst();

        assert(src != dst);

        const MfMoveFlag flag = getFlag();

        if (isKsCastling() || isQsCastling()) {
            assert(src == (whiteToMove ? Square::E1 : Square::E8));
        }

        if (isKsCastling()) {
            assert(dst == (whiteToMove ? Square::G1 : Square::G8));
        }

        if (isQsCastling()) {
            assert(dst == (whiteToMove ? Square::C1 : Square::C8));
        }

        // No pawns in backranks
        if (pt == PieceType::Pawn) {
            assert(!isBackrank(rankOf(src)));
        }

        if (flag == MfMoveFlag::PawnDoublePush) {
            assert(rankOf(src) == (whiteToMove ? Rank::Rank2 : Rank::Rank7));
            assert(rankOf(dst) == (whiteToMove ? Rank::Rank4 : Rank::Rank5));
            assert(pt == PieceType::Pawn);
        }

        // Pawn can only promote in backrank
        if (isPromo()) {
            assert(rankOf(src) == (whiteToMove ? Rank::Rank7 : Rank::Rank2));
            assert(rankOf(dst) == (whiteToMove ? Rank::Rank8 : Rank::Rank1));
            assert(pt == PieceType::Pawn);
        }

        if (isEnPassant()) {
            assert(rankOf(src) == (whiteToMove ? Rank::Rank5 : Rank::Rank4));
            assert(rankOf(dst) == (whiteToMove ? Rank::Rank6 : Rank::Rank3));
            assert(pt == PieceType::Pawn);
        }
    }

};  // struct MontyformatMove
