#pragma once

// incbin fuckery
#ifdef _MSC_VER
#define STARWAY_MSVC
#pragma push_macro("_MSC_VER")
#undef _MSC_VER
#endif

#include "../chess/montyformat_move.hpp"
#include "../chess/types.hpp"
#include "../chess/util.hpp"
#include "../incbin.h"
#include "../utils.hpp"

INCBIN(MovesMap, "moves_map_1880.bin");

// [moveSrc][moveDst][promoPieceType else 6]
const MultiArray<i16, 64, 64, 7> MOVES_MAP =
    *reinterpret_cast<const MultiArray<i16, 64, 64, 7>*>(gMovesMapData);

// The move should be flipped vertically if black to move
constexpr size_t mapMoveIdx(const MontyformatMove moveOriented) {
    const Square src = moveOriented.getSrc();
    const Square dst = moveOriented.getDst();

    const size_t thirdIdx =
        moveOriented.isPromo() ? static_cast<size_t>(moveOriented.getPromoPt().value()) : 6;

    const i16 idx = MOVES_MAP[static_cast<size_t>(src)][static_cast<size_t>(dst)][thirdIdx];
    assert(idx >= 0);

    // No black castling since move is oriented
    assert((!moveOriented.isKsCastling() && !moveOriented.isQsCastling()) || src == Square::E1);

    if (moveOriented.isQsCastling()) {
        return 1880;
    }

    if (moveOriented.isKsCastling()) {
        return 1881;
    }

    return static_cast<size_t>(idx);
}
