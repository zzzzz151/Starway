// clang-format off

#pragma once

#include "utils.hpp"

// incbin fuckery
#ifdef _MSC_VER
    #define STARWAY_MSVC
    #pragma push_macro("_MSC_VER")
    #undef _MSC_VER
#endif

#include "incbin.h"

#include <array>
#include <cassert>

INCBIN(MovesMap1880, "moves_map_1880.bin");

const std::array<std::array<std::array<i16, 7>, 64>, 64> MOVES_MAP_1880 =
    *reinterpret_cast<const std::array<std::array<std::array<i16, 7>, 64>, 64>*>(gMovesMap1880Data);

constexpr u8 moveSrc(const u16 move) { return move & 0b111'111; }

constexpr u8 moveDst(const u16 move) { return (move >> 6) & 0b111'111; }

constexpr bool isPromo(const u16 move) { return (move >> 15) > 0; }

constexpr u8 getPieceTypeMoving(const u16 move) {
    return isPromo(move) ? 0 : (move >> 12) & 0b111;
}

constexpr u8 getPromotionPieceType(const u16 move) {
    return isPromo(move) ? (move >> 12) & 0b111 : 6;
}

constexpr u16 moveAsU16(
    const u8 src, const u8 dst, const u8 pieceTypeMoving, const u8 promoPieceType = 6)
{
    assert(src < 64);
    assert(dst < 64);
    assert(pieceTypeMoving < 6);
    assert(promoPieceType != 0 && promoPieceType != 5 && promoPieceType <= 6);
    assert(promoPieceType == 6 || pieceTypeMoving == 0);

    u16 move = static_cast<u16>(src);
    move |= static_cast<u16>(dst) << 6;

    move |= static_cast<u16>(
        static_cast<u16>(promoPieceType == 6 ? pieceTypeMoving : promoPieceType) << 12
    );

    if (promoPieceType != 6)
        move |= static_cast<u16>(1) << 15;

    assert(moveSrc(move) == src);
    assert(moveDst(move) == dst);
    assert(getPieceTypeMoving(move) == pieceTypeMoving);
    assert(getPromotionPieceType(move) == promoPieceType);

    return move;
}

constexpr i16 getMoveIdx1882(
    const u8 src, const u8 dst, const u8 pieceTypeMoving, const u8 promoPieceType)
{
    assert(src < 64);
    assert(dst < 64);
    assert(pieceTypeMoving < 6);
    assert(promoPieceType != 0 && promoPieceType != 5 && promoPieceType <= 6);
    assert(promoPieceType == 6 || pieceTypeMoving == 0);

    // Assert no black castling (white is always side to move after position flip)
    assert(!(src == 60 && pieceTypeMoving == 5) ||
        (dst == 51 || dst == 52 || dst == 53 || dst == 59 || dst == 61));

    assert(MOVES_MAP_1880[src][dst][promoPieceType] >= 0);

    // Castling?
    if (src == 4 && pieceTypeMoving == 5) {
        if (dst <= 2)
            return 1880;
        if (dst >= 6)
            return 1881;
    }

    return MOVES_MAP_1880[src][dst][promoPieceType];
}
