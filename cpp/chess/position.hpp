#pragma once

#include <cassert>
#include <iostream>
#include <optional>
#include <string>

#include "../utils.hpp"
#include "attacks.hpp"
#include "montyformat_move.hpp"
#include "types.hpp"
#include "util.hpp"

struct Position {
   private:
    std::array<std::optional<PieceType>, 64> mMailbox;
    std::array<u64, 2> mColorBbs;
    std::array<u64, 6> mPieceBbs;
    u64 mCastlingRights;
    std::optional<Square> mEpSquare;
    u8 mHalfMoveClock;
    u16 mFullMoveCounter;

   public:
    Color mSideToMove;

    constexpr Position() {}  // Does not init fields

    constexpr void reset() {
        mSideToMove = Color::White;
        mMailbox = {};
        mColorBbs = {};
        mPieceBbs = {};
        mCastlingRights = 0;
        mEpSquare = std::nullopt;
        mHalfMoveClock = 0;
        mFullMoveCounter = 1;
    }

    constexpr Position(std::string fen) {
        trim(fen);
        std::vector<std::string> fenSplit = split(fen, ' ');

        if (fenSplit.size() < 4) {
            std::cerr << "Invalid fen '" << fen << "'" << std::endl;
            exit(1);
        }

        reset();

        mSideToMove = fenSplit[1] == "b" || fenSplit[1] == "B" ? Color::Black : Color::White;

        // Parse pieces
        // Iterate ranks from top to bottom, files from left to right
        i32 currentRank = 7;
        i32 currentFile = 0;
        for (char thisChar : fenSplit[0]) {
            if (thisChar == '/') {
                assert(currentRank > 0);
                currentRank--;
                currentFile = 0;
            } else if (isdigit(thisChar)) {
                currentFile += charToI32(thisChar);
                currentFile = std::min<i32>(currentFile, 7);
            } else {
                const Square square =
                    toSquare(static_cast<File>(currentFile), static_cast<Rank>(currentRank));

                const Color pieceColor = isupper(thisChar) ? Color::White : Color::Black;

                thisChar = static_cast<char>(std::tolower(thisChar));

                const PieceType pt = thisChar == 'p'   ? PieceType::Pawn
                                     : thisChar == 'n' ? PieceType::Knight
                                     : thisChar == 'b' ? PieceType::Bishop
                                     : thisChar == 'r' ? PieceType::Rook
                                     : thisChar == 'q' ? PieceType::Queen
                                                       : PieceType::King;

                togglePiece(pieceColor, pt, square);
                currentFile = std::min<i32>(currentFile + 1, 7);
            }
        }

        // Parse castling rights
        if (fenSplit[2] != "-") {
            for (const char thisChar : fenSplit[2]) {
                const Color color = isupper(thisChar) ? Color::White : Color::Black;
                const bool kingSide = thisChar == 'K' || thisChar == 'k';
                enableCastlingRight(color, kingSide);
            }
        }

        // Parse en passant square
        if (fenSplit[3] != "-") {
            mEpSquare = toSquare(fenSplit[3]);
        }

        // Parse halfmove clock
        if (fenSplit.size() > 4) {
            mHalfMoveClock = static_cast<u8>(stoi(fenSplit[4]));
        }

        // Parse full move counter
        if (fenSplit.size() > 5) {
            mFullMoveCounter = static_cast<u16>(stoi(fenSplit[5]));
        }
    }

    constexpr std::optional<PieceType> at(const Square sq) const {
        return mMailbox[static_cast<size_t>(sq)];
    }

    constexpr u64 getBb(const Color color) const { return mColorBbs[static_cast<size_t>(color)]; }

    constexpr u64 getBb(const PieceType pt) const { return mPieceBbs[static_cast<size_t>(pt)]; }

    constexpr u64 getBb(const Color color, const PieceType pt) const {
        return getBb(color) & getBb(pt);
    }

    constexpr u64 getOcc() const { return getBb(Color::White) | getBb(Color::Black); }

    constexpr Square getKingSq(const Color color) const {
        return lsb(getBb(color, PieceType::King));
    }

    constexpr bool hasCastlingRight(const Color color, const bool kingSide) const {
        if (color == Color::White) {
            return bbContainsSq(mCastlingRights, kingSide ? Square::H1 : Square::A1);
        } else {
            return bbContainsSq(mCastlingRights, kingSide ? Square::H8 : Square::A8);
        }
    }

    constexpr void enableCastlingRight(const Color color, const bool kingSide) {
        assert(getKingSq(color) == maybeRankFlipped(Square::E1, color));

        Square rookSq = maybeRankFlipped(Square::H1, color);

        if (!kingSide) {
            rookSq = fileFlipped(rookSq);
        }

        assert(bbContainsSq(getBb(color, PieceType::Rook), rookSq));
        mCastlingRights |= sqToBb(rookSq);
    }

    constexpr std::optional<Square> getEpSquare() const { return mEpSquare; }

    constexpr void setEpSquare(const std::optional<Square> newEpSq) {
        assert(!newEpSq.has_value() ||
               rankOf(*newEpSq) == (mSideToMove == Color::White ? Rank::Rank6 : Rank::Rank3));

        mEpSquare = newEpSq;
    }

    constexpr u32 getHalfMoveClock() const { return mHalfMoveClock; }

    constexpr void setHalfMoveClock(const u8 value) {
        assert(value <= 100);
        mHalfMoveClock = value;
    }

    constexpr u32 getFullMoveCounter() const { return mFullMoveCounter; }

    constexpr void setFullMoveCounter(const u16 value) {
        assert(value > 0);
        mFullMoveCounter = value;
    }

    constexpr bool isInsufficientMaterial() const {
        const auto numPieces = std::popcount(getOcc());

        if (numPieces <= 2) {
            return true;
        }

        const auto wKnightsBishopsCount = std::popcount(getBb(Color::White, PieceType::Knight) |
                                                        getBb(Color::White, PieceType::Bishop));

        const auto bKnightsBishopsCount = std::popcount(getBb(Color::Black, PieceType::Knight) |
                                                        getBb(Color::Black, PieceType::Bishop));

        if (numPieces == 3 && wKnightsBishopsCount + bKnightsBishopsCount == 1) {
            return true;
        }

        return numPieces == 4 && wKnightsBishopsCount == 1 && bKnightsBishopsCount == 1;
    }

    constexpr u64 getCheckers() const {
        const u64 checkers = getBb(!mSideToMove) & getAttackers(getKingSq(mSideToMove), getOcc());
        assert(std::popcount(checkers) <= 2);
        return checkers;
    }

    constexpr u64 getAttacks(const Color color, const u64 occ) const {
        u64 attacks = 0;

        // Pawns
        u64 pawns = getBb(color, PieceType::Pawn);
        while (pawns) {
            const Square sq = popLsb(pawns);
            attacks |= PAWN_ATTACKS[static_cast<size_t>(color)][static_cast<size_t>(sq)];
        }

        // Knights
        u64 knights = getBb(color, PieceType::Knight);
        while (knights) {
            const Square sq = popLsb(knights);
            attacks |= KNIGHT_ATTACKS[static_cast<size_t>(sq)];
        }

        // Bishops and queens
        u64 bishopsQueens = getBb(color, PieceType::Bishop) | getBb(color, PieceType::Queen);
        while (bishopsQueens) {
            const Square sq = popLsb(bishopsQueens);
            attacks |= BISHOP_ATTACKS[static_cast<size_t>(sq)].attacks(occ);
        }

        // Rooks and queens
        u64 rooksQueens = getBb(color, PieceType::Rook) | getBb(color, PieceType::Queen);
        while (rooksQueens) {
            const Square sq = popLsb(rooksQueens);
            attacks |= ROOK_ATTACKS[static_cast<size_t>(sq)].attacks(occ);
        }

        // King
        attacks |= KING_ATTACKS[static_cast<size_t>(getKingSq(color))];

        assert(attacks > 0);
        return attacks;
    }

    constexpr u64 getAttackers(const Square sq, const u64 occ) const {
        u64 attackers = 0;

        // White pawns
        attackers |= getBb(Color::White, PieceType::Pawn) &
                     PAWN_ATTACKS[static_cast<size_t>(Color::Black)][static_cast<size_t>(sq)];

        // Black pawns
        attackers |= getBb(Color::Black, PieceType::Pawn) &
                     PAWN_ATTACKS[static_cast<size_t>(Color::White)][static_cast<size_t>(sq)];

        // Knights
        attackers |= getBb(PieceType::Knight) & KNIGHT_ATTACKS[static_cast<size_t>(sq)];

        // Bishops and queens
        const u64 bishopsQueens = getBb(PieceType::Bishop) | getBb(PieceType::Queen);
        attackers |= bishopsQueens & BISHOP_ATTACKS[static_cast<size_t>(sq)].attacks(occ);

        // Rooks and queens
        const u64 rooksQueens = getBb(PieceType::Rook) | getBb(PieceType::Queen);
        attackers |= rooksQueens & ROOK_ATTACKS[static_cast<size_t>(sq)].attacks(occ);

        // King
        attackers |= getBb(PieceType::King) & KING_ATTACKS[static_cast<size_t>(sq)];

        return attackers;
    }

    // Returns pinned orthogonal and pinned diagonal
    constexpr std::pair<u64, u64> getPinned() const {
        // Bitboards to be calculated and returned
        u64 pinnedOrthogonal = 0;
        u64 pinnedDiagonal = 0;

        // Calculate pinnedOrthogonal

        const size_t ourKingSq = static_cast<size_t>(getKingSq(mSideToMove));
        const u64 occ = getOcc();

        const u64 rooksQueens = getBb(PieceType::Rook) | getBb(PieceType::Queen);
        const u64 rookAtks = ROOK_ATTACKS[ourKingSq].attacks(occ);
        const u64 newOcc = occ ^ (rookAtks & getBb(mSideToMove));
        const u64 xrayRook = rookAtks ^ ROOK_ATTACKS[ourKingSq].attacks(newOcc);

        u64 pinnersOrthogonal = getBb(!mSideToMove) & rooksQueens & xrayRook;
        while (pinnersOrthogonal > 0) {
            const Square pinnerSq = popLsb(pinnersOrthogonal);

            pinnedOrthogonal |=
                getBb(mSideToMove) & BETWEEN_EXCLUSIVE_BB[ourKingSq][static_cast<size_t>(pinnerSq)];
        }

        // Calculate pinnedDiagonal

        const u64 bishopsQueens = getBb(PieceType::Bishop) | getBb(PieceType::Queen);
        const u64 bishopAtks = BISHOP_ATTACKS[ourKingSq].attacks(occ);
        const u64 newOcc2 = occ ^ (bishopAtks & getBb(mSideToMove));
        const u64 xrayBishop = bishopAtks ^ BISHOP_ATTACKS[ourKingSq].attacks(newOcc2);

        u64 pinnersDiagonal = getBb(!mSideToMove) & bishopsQueens & xrayBishop;
        while (pinnersDiagonal > 0) {
            const Square pinnerSq = popLsb(pinnersDiagonal);

            pinnedDiagonal |=
                getBb(mSideToMove) & BETWEEN_EXCLUSIVE_BB[ourKingSq][static_cast<size_t>(pinnerSq)];
        }

        return {pinnedOrthogonal, pinnedDiagonal};
    }

    constexpr void togglePiece(const Color color, const PieceType pt, const Square sq) {
        if (mMailbox[static_cast<size_t>(sq)].has_value()) {
            assert(bbContainsSq(getBb(color, pt), sq));
            mMailbox[static_cast<size_t>(sq)] = std::nullopt;
        } else {
            assert(!bbContainsSq(getBb(color, pt), sq));
            mMailbox[static_cast<size_t>(sq)] = pt;
        }

        mColorBbs[static_cast<size_t>(color)] ^= sqToBb(sq);
        mPieceBbs[static_cast<size_t>(pt)] ^= sqToBb(sq);
    }

    constexpr void makeMove(const MontyformatMove move) {
        const Square src = move.getSrc();
        const Square dst = move.getDst();
        const std::optional<PieceType> promoPt = move.getPromoPt();

        assert(bbContainsSq(getBb(mSideToMove), src));
        assert(!bbContainsSq(getBb(mSideToMove), dst));

        const PieceType movingPt = at(src).value();

        togglePiece(mSideToMove, movingPt, src);

        if (move.isKsCastling() || move.isQsCastling()) {
            assert(hasCastlingRight(mSideToMove, move.isKsCastling()));

            Square rookSrc = move.isKsCastling() ? Square::H1 : Square::A1;
            Square rookDst = move.isKsCastling() ? Square::F1 : Square::D1;

            if (mSideToMove == Color::Black) {
                rookSrc = rankFlipped(rookSrc);
                rookDst = rankFlipped(rookDst);
            }

            assert(dst != rookSrc);
            togglePiece(mSideToMove, PieceType::King, dst);
            togglePiece(mSideToMove, PieceType::Rook, rookSrc);
            togglePiece(mSideToMove, PieceType::Rook, rookDst);
        } else if (move.isEnPassant()) {
            assert(dst == mEpSquare.value());
            assert(movingPt == PieceType::Pawn);

            const Square enemyPawnSq = enPassantRelative(dst);
            assert(bbContainsSq(getBb(!mSideToMove, PieceType::Pawn), enemyPawnSq));

            togglePiece(!mSideToMove, PieceType::Pawn, enemyPawnSq);
            togglePiece(mSideToMove, PieceType::Pawn, dst);
        } else {
            const PieceType placedPt = promoPt.value_or(movingPt);
            const std::optional<PieceType> victimPt = at(dst);

            if (move.isCapture()) {
                togglePiece(!mSideToMove, victimPt.value(), dst);
            } else {
                assert(!victimPt.has_value());
                assert(!bbContainsSq(getOcc(), dst));
            }

            togglePiece(mSideToMove, placedPt, dst);
        }

        if (movingPt == PieceType::King) {
            mCastlingRights &= ~sqToBb(maybeRankFlipped(Square::A1, mSideToMove));
            mCastlingRights &= ~sqToBb(maybeRankFlipped(Square::H1, mSideToMove));
        } else if (bbContainsSq(mCastlingRights, src)) {
            mCastlingRights &= ~sqToBb(src);
        }

        if (bbContainsSq(mCastlingRights, dst)) {
            mCastlingRights &= ~sqToBb(dst);
        }

        mSideToMove = !mSideToMove;

        if (move.isPawnDoublePush()) {
            mEpSquare = enPassantRelative(dst);
        } else {
            mEpSquare = std::nullopt;
        }

        if (movingPt != PieceType::Pawn && !move.isCapture()) {
            mHalfMoveClock = 0;
        } else {
            mHalfMoveClock++;
            assert(mHalfMoveClock <= 100);
        }

        if (mSideToMove == Color::White) {
            mFullMoveCounter++;
        }
    }

    constexpr void display() const {
        constexpr std::array<char, 6> PIECE_CHARS = {'P', 'N', 'B', 'R', 'Q', 'K'};

        for (i32 rankI32 = 7; rankI32 >= 0; rankI32--) {
            const Rank rank = static_cast<Rank>(rankI32);

            for (i32 fileI32 = 0; fileI32 < 8; fileI32++) {
                const File file = static_cast<File>(fileI32);
                const Square sq = toSquare(file, rank);
                const std::optional<PieceType> pt = at(sq);

                char pieceChar = '-';

                if (bbContainsSq(getOcc(), sq)) {
                    pieceChar = PIECE_CHARS[static_cast<size_t>(*pt)];
                }

                if (bbContainsSq(getBb(Color::Black, *pt), sq)) {
                    pieceChar = static_cast<char>(std::tolower(pieceChar));
                }

                if (file != File::A) {
                    std::cout << " ";
                }

                std::cout << std::string() + pieceChar;
            }

            std::cout << "\n";
        }

        std::cout << "Side to move: " << (mSideToMove == Color::White ? "White" : "Black") << "\n";
        std::cout << "Halfmove clock: " << std::to_string(mHalfMoveClock) << "\n";

        std::cout << std::flush;
    }

    constexpr void validate() const {
        // Assert mailbox matches bitboards and bitboards are disjoint
        u64 occ = 0;
        for (const Color color : {Color::White, Color::Black}) {
            for (size_t pieceType = 0; pieceType < 6; pieceType++) {
                const PieceType pt = static_cast<PieceType>(pieceType);
                u64 bb = getBb(color, pt);

                assert((occ & bb) == 0);
                occ |= bb;

                while (bb > 0) {
                    const Square sq = popLsb(bb);
                    assert(at(sq).value() == pt);
                }
            }
        }

        assert(occ == getOcc());

        // Assert valid number of pieces
        const size_t numPieces = static_cast<size_t>(std::popcount(occ));
        assert(numPieces >= 2 && numPieces <= 32);

        // Assert each color has exactly 1 king
        assert(std::popcount(getBb(Color::White, PieceType::King)) == 1);
        assert(std::popcount(getBb(Color::Black, PieceType::King)) == 1);

        if (hasCastlingRight(Color::White, true) || hasCastlingRight(Color::White, false)) {
            assert(getKingSq(Color::White) == Square::E1);
        }

        if (hasCastlingRight(Color::Black, true) || hasCastlingRight(Color::Black, false)) {
            assert(getKingSq(Color::Black) == Square::E8);
        }

        // Assert valid en passant square
        if (mEpSquare.has_value()) {
            assert(rankOf(*mEpSquare) == (mSideToMove == Color::White ? Rank::Rank6 : Rank::Rank3));
        }

        // Assert no pawns in backranks and no more than 2 checkers
        assert((getBb(PieceType::Pawn) & 0xff000000000000ffULL) == 0);
        assert(std::popcount(getCheckers()) <= 2);

        assert(mHalfMoveClock <= 100);
        assert(mFullMoveCounter > 0);
    }
} __attribute__((packed));  // struct Position
