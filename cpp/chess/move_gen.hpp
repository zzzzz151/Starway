#pragma once

#include "../array_vec.hpp"
#include "../utils.hpp"
#include "attacks.hpp"
#include "position.hpp"
#include "types.hpp"
#include "util.hpp"

constexpr ArrayVec<MontyformatMove, 256> getLegalMoves(const Position& pos) {
    ArrayVec<MontyformatMove, 256> moves;

    const Color stm = pos.mSideToMove;
    const Square ourKingSq = pos.getKingSq(stm);
    const u64 occ = pos.getOcc();
    const u64 us = pos.getBb(stm);
    const u64 them = pos.getBb(!stm);
    const u64 enemyAtks = pos.getAttacks(!stm, occ ^ sqToBb(ourKingSq));

    // King moves
    u64 kingAttacks = KING_ATTACKS[static_cast<size_t>(ourKingSq)] & ~us & ~enemyAtks;
    while (kingAttacks > 0) {
        const Square dst = popLsb(kingAttacks);
        const MfMoveFlag flag = bbContainsSq(occ, dst) ? MfMoveFlag::Capture : MfMoveFlag::Quiet;
        moves.pushBack(MontyformatMove(ourKingSq, dst, flag));
    }

    // If 2 checkers, only king moves are legal
    const u64 checkers = pos.getCheckers();
    if (std::popcount(checkers) > 1) {
        return moves;
    }

    // Castling
    if (ourKingSq == maybeRankFlipped(Square::E1, stm) && checkers == 0) {
        // King side castling
        if (pos.hasCastlingRight(stm, true)) {
            const Square kingDst = maybeRankFlipped(Square::G1, stm);
            const Square rookSrc = maybeRankFlipped(Square::H1, stm);

            const u64 btwnExcl =
                BETWEEN_EXCLUSIVE_BB[static_cast<size_t>(ourKingSq)][static_cast<size_t>(rookSrc)];

            if (((occ | enemyAtks) & btwnExcl) == 0) {
                moves.pushBack(MontyformatMove(ourKingSq, kingDst, MfMoveFlag::CastlingKs));
            }
        }

        // Queen side castling
        if (pos.hasCastlingRight(stm, false)) {
            const Square kingDst = maybeRankFlipped(Square::C1, stm);
            const Square rookSrc = maybeRankFlipped(Square::A1, stm);
            const Square rookDst = maybeRankFlipped(Square::D1, stm);

            const u64 btwnExcl =
                BETWEEN_EXCLUSIVE_BB[static_cast<size_t>(ourKingSq)][static_cast<size_t>(rookSrc)];

            if ((occ & btwnExcl) == 0 && !bbContainsSq(enemyAtks, kingDst) &&
                !bbContainsSq(enemyAtks, rookDst)) {
                moves.pushBack(MontyformatMove(ourKingSq, kingDst, MfMoveFlag::CastlingQs));
            }
        }
    }

    // Movable squares mask
    u64 movableBb = ~0ULL;
    if (checkers > 0) {
        const Square checkerSq = lsb(checkers);

        const u64 sliders =
            pos.getBb(PieceType::Bishop) | pos.getBb(PieceType::Rook) | pos.getBb(PieceType::Queen);

        movableBb = checkers;

        if (bbContainsSq(sliders, checkerSq)) {
            movableBb |= BETWEEN_EXCLUSIVE_BB[static_cast<size_t>(ourKingSq)]
                                             [static_cast<size_t>(checkerSq)];
        }
    }

    const auto [pinnedOrthogonal, pinnedDiagonal] = pos.getPinned();
    const u64 pinnedBb = pinnedOrthogonal | pinnedDiagonal;

    // Pawns moves

    const auto pushPawnMoveMaybePromos = [&](const Square src, const Square dst) {
        const bool isCapture = bbContainsSq(occ, dst);

        if (!isBackrank(rankOf(dst))) {
            const auto flag = isCapture ? MfMoveFlag::Capture : MfMoveFlag::Quiet;
            moves.pushBack(MontyformatMove(src, dst, flag));
            return;
        }

        // Promotion

        const u16 baseFlag =
            static_cast<u16>(isCapture ? MfMoveFlag::KnightPromoCapture : MfMoveFlag::KnightPromo);

        for (size_t i = 0; i < 4; i++) {
            moves.pushBack(MontyformatMove(src, dst, static_cast<MfMoveFlag>(baseFlag + i)));
        }
    };

    u64 pawnsBb = pos.getBb(stm, PieceType::Pawn);
    while (pawnsBb > 0) {
        const Square src = popLsb(pawnsBb);
        assert(!isBackrank(rankOf(src)));

        // Pawn's captures

        u64 pawnCaptures =
            PAWN_ATTACKS[static_cast<size_t>(stm)][static_cast<size_t>(src)] & movableBb & them;

        if (bbContainsSq(pinnedBb, src)) {
            pawnCaptures &= LINE_THRU_BB[static_cast<size_t>(ourKingSq)][static_cast<size_t>(src)];
        }

        while (pawnCaptures > 0) {
            const Square dst = popLsb(pawnCaptures);
            pushPawnMoveMaybePromos(src, dst);
        }

        // Single and double pushes

        if (bbContainsSq(pinnedDiagonal, src)) {
            continue;
        }

        u64 pinRay = LINE_THRU_BB[static_cast<size_t>(ourKingSq)][static_cast<size_t>(src)];
        pinRay &= pinRay << 1;

        // Pawn pinnedBb horizontally?
        if (bbContainsSq(pinnedOrthogonal, src) && pinRay > 0) {
            continue;
        }

        const Square singlePushDst =
            static_cast<Square>(static_cast<i32>(src) + (stm == Color::White ? 8 : -8));

        if (bbContainsSq(occ, singlePushDst)) {
            continue;
        }

        if (bbContainsSq(movableBb, singlePushDst)) {
            pushPawnMoveMaybePromos(src, singlePushDst);
        }

        // If pawn has moved, skip
        if (rankOf(src) != (stm == Color::White ? Rank::Rank2 : Rank::Rank7)) {
            continue;
        }

        const Square doublePushDst =
            static_cast<Square>(static_cast<i32>(src) + (stm == Color::White ? 16 : -16));

        if (!bbContainsSq(occ, doublePushDst) && bbContainsSq(movableBb, doublePushDst)) {
            moves.pushBack(MontyformatMove(src, doublePushDst, MfMoveFlag::PawnDoublePush));
        }
    }

    // En passant moves
    if (pos.getEpSquare().has_value()) {
        const Square epSquare = *(pos.getEpSquare());
        const Square capturedPawnSq = enPassantRelative(epSquare);

        u64 ourEpPawns = pos.getBb(stm, PieceType::Pawn) &
                         PAWN_ATTACKS[static_cast<size_t>(!stm)][static_cast<size_t>(epSquare)];

        while (ourEpPawns > 0) {
            const Square src = popLsb(ourEpPawns);
            const u64 occAfterEp = occ ^ sqToBb(src) ^ sqToBb(capturedPawnSq) ^ sqToBb(epSquare);
            const u64 bishopsQueens = pos.getBb(PieceType::Bishop) | pos.getBb(PieceType::Queen);
            const u64 rooksQueens = pos.getBb(PieceType::Rook) | pos.getBb(PieceType::Queen);

            u64 sliderAttackers =
                bishopsQueens & BISHOP_ATTACKS[static_cast<size_t>(ourKingSq)].attacks(occAfterEp);

            sliderAttackers |=
                rooksQueens & ROOK_ATTACKS[static_cast<size_t>(ourKingSq)].attacks(occAfterEp);

            if ((them & sliderAttackers) == 0) {
                moves.pushBack(MontyformatMove(src, epSquare, MfMoveFlag::EnPassant));
            }
        }
    }

    const u64 mask = ~us & movableBb;

    // Knights moves
    u64 ourKnights = pos.getBb(stm, PieceType::Knight) & ~pinnedBb;
    while (ourKnights > 0) {
        const Square src = popLsb(ourKnights);
        u64 knightMoves = KNIGHT_ATTACKS[static_cast<size_t>(src)] & mask;

        while (knightMoves > 0) {
            const Square dst = popLsb(knightMoves);
            const auto flag = bbContainsSq(occ, dst) ? MfMoveFlag::Capture : MfMoveFlag::Quiet;
            moves.pushBack(MontyformatMove(src, dst, flag));
        }
    }

    // Bishops moves
    u64 ourBishops = pos.getBb(stm, PieceType::Bishop) & ~pinnedOrthogonal;
    while (ourBishops > 0) {
        const Square src = popLsb(ourBishops);
        u64 bishopMoves = BISHOP_ATTACKS[static_cast<size_t>(src)].attacks(occ) & mask;

        if (bbContainsSq(pinnedDiagonal, src)) {
            bishopMoves &= LINE_THRU_BB[static_cast<size_t>(ourKingSq)][static_cast<size_t>(src)];
        }

        while (bishopMoves > 0) {
            const Square dst = popLsb(bishopMoves);
            const auto flag = bbContainsSq(occ, dst) ? MfMoveFlag::Capture : MfMoveFlag::Quiet;
            moves.pushBack(MontyformatMove(src, dst, flag));
        }
    }

    // Rooks moves
    u64 ourRooks = pos.getBb(stm, PieceType::Rook) & ~pinnedDiagonal;
    while (ourRooks > 0) {
        const Square src = popLsb(ourRooks);
        u64 rookMoves = ROOK_ATTACKS[static_cast<size_t>(src)].attacks(occ) & mask;

        if (bbContainsSq(pinnedOrthogonal, src)) {
            rookMoves &= LINE_THRU_BB[static_cast<size_t>(ourKingSq)][static_cast<size_t>(src)];
        }

        while (rookMoves > 0) {
            const Square dst = popLsb(rookMoves);
            const auto flag = bbContainsSq(occ, dst) ? MfMoveFlag::Capture : MfMoveFlag::Quiet;
            moves.pushBack(MontyformatMove(src, dst, flag));
        }
    }

    // Queens moves
    u64 ourQueens = pos.getBb(stm, PieceType::Queen);
    while (ourQueens > 0) {
        const Square src = popLsb(ourQueens);
        u64 queenMoves = getQueenAttacks(src, occ) & mask;

        if (bbContainsSq(pinnedBb, src)) {
            queenMoves &= LINE_THRU_BB[static_cast<size_t>(ourKingSq)][static_cast<size_t>(src)];
        }

        while (queenMoves > 0) {
            const Square dst = popLsb(queenMoves);
            const auto flag = bbContainsSq(occ, dst) ? MfMoveFlag::Capture : MfMoveFlag::Quiet;
            moves.pushBack(MontyformatMove(src, dst, flag));
        }
    }

    return moves;
}
