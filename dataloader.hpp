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

    bool isWhiteStm;

    u64 occupied;

    // 4 bits per piece for a max of 32 pieces
    // lsb of the 4 bits is piece color, other 3 bits is piece type
    u128 pieces;

    u8 whiteKingSquare, blackKingSquare, whiteQueenSquare, blackQueenSquare;

    i16 stmScore;

    i8 stmWdl; // -1, 0, 1 for loss, draw, win, respectively

} __attribute__((packed)); // struct DataEntry

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
