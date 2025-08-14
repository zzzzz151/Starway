// clang-format off

#pragma once

#include <cstdint>
#include <bit>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

constexpr auto lsb(const u64 bitboard) {
    return std::countr_zero(bitboard);
}

constexpr auto popLsb(u64& bitboard) {
    const auto idx = lsb(bitboard);
    bitboard &= bitboard - 1; // compiler optimizes this to _blsr_u64
    return idx;
}
