#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <optional>

#include "../chess/position.hpp"
#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../utils.hpp"

// Montyformat compressed board
// https://github.com/JonathanHallstrom/montyformat/blob/main/docs/basic_layout.md#compressed-board
struct CompressedBoard {
   private:
    std::array<u64, 4> mBbs;
    u8 mStm;  // 0 = white, 1 = black
    u8 mEpSquare;
    u8 mCastlingRights;
    u8 mHalfMoveClock;
    u16 mFullMoveCounter;
    std::array<u8, 4> mCastlingFiles;

   public:
    constexpr CompressedBoard() {}  // Does not init fields

    constexpr Color sideToMove() const {
        assert(mStm < 2);
        return static_cast<Color>(mStm);
    }

    constexpr u64 getOcc() const { return mBbs[1] | mBbs[2] | mBbs[3]; }

    // White, black
    constexpr std::array<u64, 2> getColorBbs() const { return {getOcc() ^ mBbs[0], mBbs[0]}; }

    constexpr std::array<u64, 6> getPieceBbs() const {
        const u64 bishops = mBbs[2] & mBbs[3];
        const u64 queens = mBbs[1] & mBbs[3];
        const u64 kings = mBbs[1] & mBbs[2];
        const u64 pawns = mBbs[3] ^ bishops ^ queens;
        const u64 knights = mBbs[2] ^ bishops ^ kings;
        const u64 rooks = mBbs[1] ^ kings ^ queens;

        return {pawns, knights, bishops, rooks, queens, kings};
    }

    constexpr bool isFrc() const {
        for (const u8 rookFile : mCastlingFiles) {
            assert(rookFile < 8);

            if (rookFile != 0 && rookFile != 7) {
                return true;
            }
        }

        return false;
    }

    constexpr Position decompress() const {
        assert(!isFrc());

        Position pos;
        pos.reset();

        pos.mSideToMove = sideToMove();

        const std::array<u64, 2> colorBbs = getColorBbs();
        const std::array<u64, 6> pieceBbs = getPieceBbs();

        for (const Color color : {Color::White, Color::Black}) {
            for (size_t pieceType = 0; pieceType < 6; pieceType++) {
                const PieceType pt = static_cast<PieceType>(pieceType);
                u64 bb = colorBbs[static_cast<size_t>(color)] & pieceBbs[pieceType];

                while (bb > 0) {
                    const Square sq = popLsb(bb);
                    pos.togglePiece(color, pt, sq);
                }
            }
        }

        assert((mCastlingRights & 0b1111'0000) == 0);

        // White king side right
        if ((mCastlingRights & 0b0000'0100) > 0) {
            pos.enableCastlingRight(Color::White, true);
        }

        // White queen side right
        if ((mCastlingRights & 0b0000'1000) > 0) {
            pos.enableCastlingRight(Color::White, false);
        }

        // Black king side right
        if ((mCastlingRights & 0b0000'0001) > 0) {
            pos.enableCastlingRight(Color::Black, true);
        }

        // Black queen side right
        if ((mCastlingRights & 0b0000'0010) > 0) {
            pos.enableCastlingRight(Color::Black, false);
        }

        assert(mEpSquare <= 64);

        if (mEpSquare > 0 && mEpSquare < 64) {
            pos.setEpSquare(static_cast<Square>(mEpSquare));
        }

        pos.setHalfMoveClock(mHalfMoveClock);
        pos.setFullMoveCounter(mFullMoveCounter);

        return pos;
    }

} __attribute__((packed));  // struct CompressedBoard

static_assert(sizeof(CompressedBoard) == 8 * 4 + 1 + 1 + 1 + 1 + 2 + 4);
