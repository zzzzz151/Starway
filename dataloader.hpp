// clang-format off

#pragma once

#include <cstdint>
#include <bit>

// Needed to export functions on Windows
#ifdef _WIN32
    #define API __declspec(dllexport)
#else
    #define API
#endif

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

constexpr auto lsb(const u64 bitboard)
{
    return std::countr_zero(bitboard);
}

constexpr auto popLsb(u64& bitboard)
{
    const auto idx = lsb(bitboard);
    bitboard &= bitboard - 1; // compiler optimizes this to _blsr_u64
    return idx;
}

struct DataEntry
{
public:

    // Lsb is set if black to move
    // Highest 7 bits are halfmove clock
    u8 stmAndHalfmoveClock;

    u64 occupied;

    // 4 bits per piece for a max of 32 pieces
    // Lsb is set if piece color is black, other 3 bits is piece type (0-5 including both)
    u128 pieces;

    // Kings and queens squares (64 if 0 colored queens, 65 if >1 colored queens)
    u8 wkSq, bkSq, wqSq, bqSq;

    i16 stmScore;
    i8 stmResult; // -1, 0, 1

} __attribute__((packed));

static_assert(sizeof(DataEntry) == 32); // 32 bytes

struct Batch
{
public:

    // Indices of active features (a position has max 32 active features)
    i32* activeFeaturesWhite;
    i32* activeFeaturesBlack;

    bool* isWhiteStm;

    i16* stmScores;
    float* stmWDLs;

    constexpr Batch(const std::size_t batchSize)
    {
        // Indices of active features (a position has max 32 active features)
        activeFeaturesWhite = new i32[batchSize * 32];
        activeFeaturesBlack = new i32[batchSize * 32];

        isWhiteStm = new bool[batchSize];

        stmScores = new i16[batchSize];
        stmWDLs   = new float[batchSize];
    }

}; // struct Batch
