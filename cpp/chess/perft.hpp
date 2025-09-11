#pragma once

#include "../utils.hpp"
#include "attacks.hpp"
#include "move_gen.hpp"
#include "position.hpp"
#include "types.hpp"
#include "util.hpp"

constexpr u64 perft(const Position& pos, const i32 depth) {
    if (depth <= 0) {
        return 1;
    }

    const auto legalMoves = getLegalMoves(pos);

    if (depth == 1) {
        return legalMoves.size();
    }

    u64 nodes = 0;

    for (const MontyformatMove move : legalMoves) {
        Position newPos = pos;
        newPos.makeMove(move);
        nodes += perft(newPos, depth - 1);
    }

    return nodes;
}
