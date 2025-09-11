#include <cassert>

#include "../utils.hpp"
#include "perft.hpp"
#include "position.hpp"
#include "types.hpp"
#include "util.hpp"

int main() {
    // https:www.chessprogramming.org/Perft_Results

    const Position pos1Start = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    pos1Start.validate();

    const Position pos2Kiwipete =
        Position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ");

    pos2Kiwipete.validate();

    const Position pos3 = Position("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    pos3.validate();

    const Position pos4 =
        Position("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");

    pos4.validate();

    const Position pos4Mirrored =
        Position("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1");

    pos4Mirrored.validate();

    const Position pos5 = Position("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    pos5.validate();

    assert(perft(pos1Start, 0) == 1ULL);
    assert(perft(pos1Start, -1) == 1ULL);

    // perft(1)
    assert(perft(pos1Start, 1) == 20ULL);
    assert(perft(pos2Kiwipete, 1) == 48ULL);
    assert(perft(pos3, 1) == 14ULL);
    assert(perft(pos4, 1) == 6ULL);
    assert(perft(pos4Mirrored, 1) == 6ULL);
    assert(perft(pos5, 1) == 44ULL);

    // Start pos perft's
    assert(perft(pos1Start, 2) == 400ULL);
    assert(perft(pos1Start, 3) == 8902ULL);
    assert(perft(pos1Start, 4) == 197281ULL);
    assert(perft(pos1Start, 5) == 4865609ULL);
    assert(perft(pos1Start, 6) == 119060324ULL);

    // Kiwipete perft(5)
    assert(perft(pos2Kiwipete, 5) == 193690690ULL);

    std::cout << "Passed!" << std::endl;
    return 0;
}
