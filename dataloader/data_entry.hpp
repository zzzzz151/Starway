// clang-format off

#pragma once

#include "utils.hpp"
#include <cassert>
#include <bit>
#include <tuple>
#include <optional>

struct DataEntry {
private:

    // Encoding from lowest to highest bits
    // "x-y" includes both x-th and y-th bits
    // 1st:   Set if black to move
    // 2:     Set if in check
    // 3-8:   Side-to-move (always white when oriented) king square oriented
    // 9-14:  Not-side-to-move (always black when oriented) king square oriented
    // 15:    Set if stm (always white since it's oriented) short castling right
    // 16:    Set if stm (always white since it's oriented) long castling right
    // 17-20: En passant square file (8 if none)
    // 21-22: WDL (-1 if stm lost, 0 if draw, 1 if stm won)
    // 23:    Set if best move is noisy
    // 24-32: Unused
    u32 mMiscData = 0;

    u64 mOccupied = 0; // Oriented (flipped vertically if black to move)

    // 4 bits per oriented piece for a max of 32 oriented pieces
    // Lsb of the 4 bits is set if the color of the oriented piece is black
    // Other 3 bits is piece type (0-5 including both)
    u128 mPieces = 0;

    i16 mStmScore = 0;

    u16 mMove = 0; // Oriented (flipped vertically if black to move)

public:

    constexpr DataEntry() = default;

    constexpr DataEntry(
        const bool _whiteToMove,
        const bool _inCheck,
        const u8 _stmKingSqOriented,
        const u8 _ntmKingSqOriented,
        const bool _kCastleRightOriented,
        const bool _qCastleRightOriented,
        const u8 _epSquareOriented,
        const i16 _stmScore,
        const i8 _stmWdl,
        const u16 _moveOriented,
        const bool _isNoisy)
    {
        assert(_epSquareOriented <= 64);
        assert(_epSquareOriented == 64 || _epSquareOriented / 8 == 5);
        assert(_stmWdl == -1 || _stmWdl == 0 || _stmWdl == 1);

        const u8 epSqFile = _epSquareOriented == 64 ? 8 : _epSquareOriented % 8;

        mMiscData = static_cast<u32>(!_whiteToMove)
                  | (static_cast<u32>(_inCheck) << 1)
                  | (static_cast<u32>(_stmKingSqOriented) << 2)
                  | (static_cast<u32>(_ntmKingSqOriented) << 8)
                  | (static_cast<u32>(_kCastleRightOriented) << 14)
                  | (static_cast<u32>(_qCastleRightOriented) << 15)
                  | (static_cast<u32>(epSqFile) << 16)
                  | (static_cast<u32>(_stmWdl + 1) << 20)
                  | (static_cast<u32>(_isNoisy) << 22);

        assert(whiteToMove() == _whiteToMove);
        assert(inCheck() == _inCheck);
        assert(stmKingSqOriented() == _stmKingSqOriented);
        assert(ntmKingSqOriented() == _ntmKingSqOriented);
        assert(kCastleRightOriented() == _kCastleRightOriented);
        assert(qCastleRightOriented() == _qCastleRightOriented);
        assert(epSquareOriented() == _epSquareOriented);
        assert(stmWdl() == (_stmWdl < 0 ? 0.0f : _stmWdl > 0 ? 1.0f : 0.5f));
        assert(isBestMoveNoisy() == _isNoisy);

        mStmScore = _stmScore;
        mMove = _moveOriented;
    }

    constexpr bool whiteToMove() const { return (mMiscData & 0b1) == 0; }

    constexpr bool inCheck() const { return ((mMiscData >> 1) & 0b1) > 0; }

    constexpr u8 stmKingSqOriented() const { return (mMiscData >> 2) & 0b111'111; }

    constexpr u8 ntmKingSqOriented() const { return (mMiscData >> 8) & 0b111'111; }

    constexpr bool kCastleRightOriented() const { return ((mMiscData >> 14) & 0b1) > 0; }

    constexpr bool qCastleRightOriented() const { return ((mMiscData >> 15) & 0b1) > 0; }

    constexpr u8 epSquareOriented() const {
        const u8 epFile = (mMiscData >> 16) & 0b1111;
        assert(epFile <= 8);
        return epFile == 8 ? 64 : 40 + epFile;
    }

    constexpr i16 stmScore() const { return mStmScore; }

    constexpr float stmWdl() const {
        const u8 storedWdl = (mMiscData >> 20) & 0b11;
        assert(storedWdl == 0 || storedWdl == 1 || storedWdl == 2);
        return storedWdl == 0 ? 0.0f : storedWdl == 1 ? 0.5f : 1.0f;
    }

    constexpr u16 bestMoveOriented() const { return mMove; }

    constexpr bool isBestMoveNoisy() const { return ((mMiscData >> 22) & 0b1) > 0; }

    // Piece color is 0 for white and 1 for black, piece type is 0-5
    constexpr void addOrientedPiece(const u8 pieceColor, const u8 pieceType, const u8 square)
    {
        assert(pieceColor < 2 && pieceType < 6 && square < 64);
        assert(std::popcount(mOccupied) < 32);

        assert(pieceType != 5 ||
            square == (pieceColor == 0 ? stmKingSqOriented() : ntmKingSqOriented()));

        const u128 fourBitsPiece = static_cast<u128>(pieceColor | (pieceType << 1));
        mPieces |= fourBitsPiece << (std::popcount(mOccupied) * 4);
        mOccupied |= 1ULL << square;
    }

    // Returns piece color (0 for white, 1 for black), piece type (0-5), square
    constexpr std::optional<std::tuple<u8, u8, u8>> popOrientedPiece()
    {
        if (mOccupied == 0)
            return std::nullopt;

        const u8 square = static_cast<u8>(__builtin_ctzll(mOccupied));

        // Pop lsb (compiler optimizes this to _blsr_u64)
        mOccupied &= mOccupied - 1;

        const u8 pieceColor = mPieces & 0b1;
        const u8 pieceType = (mPieces & 0b1110) >> 1;

        assert(pieceType < 6);

        assert(pieceType != 5 ||
            square == (pieceColor == 0 ? stmKingSqOriented() : ntmKingSqOriented()));

        mPieces >>= 4;

        return std::optional<std::tuple<u8, u8, u8>>({ pieceColor, pieceType, square });
    }

} __attribute__((packed)); // struct DataEntry

static_assert(sizeof(DataEntry) == 32); // 32 bytes
