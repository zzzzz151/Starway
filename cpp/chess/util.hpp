#pragma once

#include <bit>
#include <cassert>
#include <iostream>
#include <string>

#include "../utils.hpp"
#include "types.hpp"

constexpr Square toSquare(const File f, const Rank r) {
    const u8 sq = static_cast<u8>(r) * 8 + static_cast<u8>(f);
    return static_cast<Square>(sq);
}

constexpr File fileOf(const Square sq) { return static_cast<File>(static_cast<u8>(sq) % 8); }

constexpr Rank rankOf(const Square sq) { return static_cast<Rank>(static_cast<u8>(sq) / 8); }

constexpr Square fileFlipped(const Square sq) {
    return static_cast<Square>(static_cast<u8>(sq) ^ 7);
}

constexpr Square rankFlipped(const Square sq) {
    return static_cast<Square>(static_cast<u8>(sq) ^ 56);
}

constexpr Square maybeRankFlipped(const Square sq, const Color sideToMove) {
    return sideToMove == Color::White ? sq : rankFlipped(sq);
}

constexpr Square enPassantRelative(const Square sq) {
    const u8 rank = static_cast<u8>(rankOf(sq));
    assert(rank >= 2 && rank <= 5);
    return static_cast<Square>(static_cast<u8>(sq) ^ 8);
}

constexpr bool isBackrank(const Rank r) { return r == Rank::Rank1 || r == Rank::Rank8; }

constexpr Square toSquare(std::string s) {
    trim(s);

    if (s.size() != 2) {
        std::cerr << "Square '" << s << "'" << " must be exactly 2 chars" << std::endl;
        exit(1);
    }

    const i32 file = static_cast<i32>(s[0]) - static_cast<i32>('a');
    const i32 rank = static_cast<i32>(s[1]) - static_cast<i32>('1');

    assert(file >= 0 && file < 8);
    assert(rank >= 0 && rank < 8);

    return toSquare(static_cast<File>(file), static_cast<Rank>(rank));
}

constexpr std::string squareToStr(const Square sq) {
    std::string res = "";
    res += static_cast<char>('a' + static_cast<char>(fileOf(sq)));
    res += static_cast<char>('1' + static_cast<char>(rankOf(sq)));
    return res;
}

constexpr u64 sqToBb(const Square sq) { return 1ULL << static_cast<u8>(sq); }

constexpr u64 fileBb(const File f) {
    constexpr u64 FILE_A_BB = 0x101010101010101ULL;
    return FILE_A_BB << static_cast<u8>(f);
}

constexpr u64 rankBb(const Rank r) {
    constexpr u64 RANK_1_BB = 0xffULL;
    return RANK_1_BB << (static_cast<u8>(r) * 8);
}

constexpr bool bbContainsSq(const u64 bb, const Square sq) { return (bb & sqToBb(sq)) > 0; }

constexpr Square lsb(const u64 bb) {
    assert(bb > 0);
    return static_cast<Square>(std::countr_zero(bb));
}

constexpr Square popLsb(u64& bb) {
    const Square sq = lsb(bb);
    bb &= bb - 1;  // Compiler optimizes this to _blsr_u64
    return sq;
}
