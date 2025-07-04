// tests.cpp
#include <gtest/gtest.h>
#include "game.hpp"
#include "player.hpp"

// Note: these tests assume that Game::gameboard is accessible for setup.
// If it's private, you'll need to add a setter or make it protected for testing.

TEST(Player_functionality, newplayer)
{
    Player p1("romanos", Player::PLAYER1);
    ASSERT_EQ(p1.getName(), "romanos");
}

// ------ ValidOrigin tests ------

TEST(ValidOrigin, test_origin_boolean)
{
    Game game(0);
    ASSERT_TRUE(game.isValidOrigin(+1, 1));
    ASSERT_FALSE(game.isValidOrigin(-1, 1));
}

TEST(ValidOrigin, test_origin_for_player2)
{
    Game game(0);
    ASSERT_FALSE(game.isValidOrigin(+1, 6));
    ASSERT_TRUE(game.isValidOrigin(-1, 6));
}

TEST(ValidOrigin, gameboard_with_no_pieces)
{
    Game game(0);
    ASSERT_FALSE(game.isValidOrigin(+1, 2));
    ASSERT_FALSE(game.isValidOrigin(-1, 2));
    ASSERT_FALSE(game.isValidOrigin(-1, 7));
}

TEST(ValidOrigin, test_p2_first_origin)
{
    Game game(1);
    ASSERT_FALSE(game.isValidOrigin(+1, 6));
    ASSERT_TRUE(game.isValidOrigin(-1, 6));
}

// ------ ValidDestination tests ------

TEST(ValidDestination, with_friendly_piece)
{
    Game game(1);
    ASSERT_TRUE(game.isValidDestination(+1, 1));
    ASSERT_TRUE(game.isValidDestination(+1, 12));
    ASSERT_TRUE(game.isValidDestination(-1, 6));
    ASSERT_TRUE(game.isValidDestination(-1, 13));
}

TEST(ValidDestination, with_no_piece)
{
    Game game(1);
    ASSERT_TRUE(game.isValidDestination(+1, 2));
    ASSERT_TRUE(game.isValidDestination(+1, 11));
    ASSERT_TRUE(game.isValidDestination(-1, 2));
    ASSERT_TRUE(game.isValidDestination(-1, 22));
}

TEST(ValidDestination, with_enemy_blockade)
{
    Game game(0);
    ASSERT_FALSE(game.isValidDestination(+1, 6));
    ASSERT_FALSE(game.isValidDestination(+1, 8));
    ASSERT_FALSE(game.isValidDestination(-1, 19));
    ASSERT_FALSE(game.isValidDestination(-1, 17));
}

// ------ CapturePiece via tryMove tests ------

TEST(CapturePiece, singleBlotIsTakenAndJailed_via_tryMove)
{
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);

    // place a single P2 blot on point 2
    game.gameboard[1] = -1;
    std::string err;

    // P1 rolls a 1 from 1→2
    bool ok = game.tryMove(&p1, 1, 1, 2, err);
    ASSERT_TRUE(ok) << "err=" << err;

    // blot captured, P1 checker sits there
    auto board = game.getGameBoard();
    EXPECT_EQ(board[1], +1);
    // P2 jailed one
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER2), 1);
}

TEST(CapturePiece, cannotCaptureTwoOrMore_via_tryMove)
{
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);
    std::string err;

    // Attempt P1 moving 5 spaces onto point 6 (blocked)
    bool ok = game.tryMove(&p1, 5, 1, 6, err);
    EXPECT_FALSE(ok);
    EXPECT_EQ(err, "Invalid destination.");
}

TEST(CapturePiece, invalidOrigin_with_jailed_piece_via_tryMove)
{
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);
    game.pieces.addJailedPiece(Player::PLAYER1);

    std::string err;
    // wrong origin
    EXPECT_FALSE(game.tryMove(&p1, 5, 12, 7, err));
    EXPECT_EQ(err, "Invalid origin");

    // correct jail exit (dice=5, origin=0→dest=5)
    // clear point 5 first
    game.gameboard[4] = 0;
    EXPECT_TRUE(game.tryMove(&p1, 5, 0, 5, err)) << err;
}

// ------ tryMove tests for moveOne behaviors ------

TEST(TryMove, FreesJailedPieceAndPlacesOnBoard)
{
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    Game game(1);
    game.setPlayers(&p1, &p2);
    game.pieces.addJailedPiece(Player::PLAYER1);

    // clear point 2
    game.gameboard[1] = 0;

    std::string err;
    bool ok = game.tryMove(&p1, 2, 0, 2, err);
    EXPECT_TRUE(ok) << err;
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
    EXPECT_EQ(game.getGameBoard()[1], +1);
}

TEST(TryMove, RejectsNonZeroOriginWhenJailed)
{
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);
    game.pieces.addJailedPiece(Player::PLAYER1);

    std::string err;
    // wrong origin
    EXPECT_FALSE(game.tryMove(&p1, 7, 5, 7, err));
    EXPECT_EQ(err, "Invalid origin");

    // correct exit to point 7
    game.gameboard[6] = 0;
    EXPECT_TRUE(game.tryMove(&p1, 7, 0, 7, err)) << err;
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
    EXPECT_EQ(game.getGameBoard()[6], +1);
}

TEST(TryMove, NormalMoveWhenNoJail)
{
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    Game game(0);
    game.setPlayers(&p1, &p2);

    // place 2 on point 3, clear point 5
    game.gameboard[2] = +2;
    game.gameboard[4] = 0;

    std::string err;
    bool ok = game.tryMove(&p1, 2, 3, 5, err);
    EXPECT_TRUE(ok) << err;

    auto b = game.getGameBoard();
    EXPECT_EQ(b[2], +1);
    EXPECT_EQ(b[4], +1);
    EXPECT_EQ(game.pieces.numJailed(Player::PLAYER1), 0);
}

// ------ over(&winner) test ------

TEST(GameOver, detects_winner)
{
    Game game(0);
    Player p1("Roma", Player::PLAYER1), p2("Romanos", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    // free 15 pieces for P2
    for (int i = 0; i < 15; ++i)
        game.pieces.freePiece(Player::PLAYER2);

    int winner = -1;
    EXPECT_TRUE(game.over(&winner));
    EXPECT_EQ(winner, Player::PLAYER2);
}
// Test that freeing a piece requires no jailed pieces and all are in home
TEST(FreeingPiece, CannotFreeWhenNotInHomeOrJailed)
{
    Game game(4);
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    // Initially, P1 has no jailed pieces but pieces are not all in home
    // Attempt to free from origin=0
    EXPECT_FALSE(game.isValidDestination(+1, 0));
    EXPECT_EQ(game.pieces.numFreed(Player::PLAYER1), 0);

    // Now simulate a jailed piece and attempt free
    game.pieces.addJailedPiece(Player::PLAYER1);
    EXPECT_FALSE(game.isValidDestination(+1, 0));
    EXPECT_EQ(game.pieces.numFreed(Player::PLAYER1), 0);
}

// Test that a single valid free increments numFreed
TEST(FreeingPiece, CanFreeWhenEligible)
{
    Game game(4);
    Player p1("Alice", Player::PLAYER1), p2("Bob", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    // Clear board and place one P1 checker in home region at point 19 (index 18)
    for (int i = 0; i < 24; ++i)
        game.gameboard[i] = 0;
    game.gameboard[18] = 1;

    // No jailed pieces, and only home region occupied
    EXPECT_TRUE(game.isValidDestination(1, 0));
    // FIXED: Validation functions should NOT modify state
    EXPECT_EQ(game.pieces.numFreed(Player::PLAYER1), 0);

    // To actually free a piece, you need to call tryMove()
    std::string err;
    bool ok = game.tryMove(&p1, 6, 19, 25, err);
    EXPECT_TRUE(ok) << err;
    EXPECT_EQ(game.pieces.numFreed(Player::PLAYER1), 1);
}

// Test that Game::over detects winner by freed count
TEST(GameOver, DetectsPlayer1Winner)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    // Simulate P1 has freed 15 pieces
    for (int i = 0; i < 15; ++i)
        game.pieces.freePiece(Player::PLAYER1);

    int winner = -1;
    EXPECT_TRUE(game.over(&winner));
    EXPECT_EQ(winner, Player::PLAYER1);
}

TEST(GameOver, DetectsPlayer2Winner)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    // Simulate P2 has freed 15 pieces
    for (int i = 0; i < 15; ++i)
        game.pieces.freePiece(Player::PLAYER2);

    int winner = -1;
    EXPECT_TRUE(game.over(&winner));
    EXPECT_EQ(winner, Player::PLAYER2);
}

// Helpers to compare sequences
static bool containsSequence(
    const vector<vector<pair<int, int>>> &seqs,
    const vector<pair<int, int>> &target)
{
    for (auto &s : seqs)
    {
        if (s == target)
            return true;
    }
    return false;
}

TEST(LegalMoves, InitialP1Die1)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    auto moves = game.legalMoves(Player::PLAYER1, 1);
    // Should have three legal single‑pip moves: 1→2, 17→18, 19→20
    ASSERT_EQ(moves.size(), 3);
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(1, 2)) != moves.end());
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(17, 18)) != moves.end());
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(19, 20)) != moves.end());
}

TEST(LegalMoves, InitialP2Die1)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    auto moves = game.legalMoves(Player::PLAYER2, 1);
    // Should have three legal single‑pip moves: 6→5, 8→7, 24→23
    ASSERT_EQ(moves.size(), 3);
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(6, 5)) != moves.end());
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(8, 7)) != moves.end());
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(24, 23)) != moves.end());
}

TEST(LegalMoves, P1Die5Blocked)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    auto moves = game.legalMoves(Player::PLAYER1, 5);
    // At launch, P1 has two legal 5‑pip moves: 12→17 and 17→22
    ASSERT_EQ(moves.size(), 2);
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(12, 17)) != moves.end());
    EXPECT_TRUE(std::find(moves.begin(), moves.end(), std::make_pair(17, 22)) != moves.end());
}

TEST(LegalMoves, JailedP1NoMoveIfBlocked)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    game.getPieces().addJailedPiece(Player::PLAYER1);
    auto moves = game.legalMoves(Player::PLAYER1, 6);
    EXPECT_TRUE(moves.empty());
}

TEST(LegalMoves, JailedP1BarExit)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    game.getPieces().addJailedPiece(Player::PLAYER1);
    auto moves = game.legalMoves(Player::PLAYER1, 5);
    ASSERT_EQ(moves.size(), 1);
    EXPECT_EQ(moves[0], make_pair(0, 5));
}

TEST(LegalTurnSeq, NonDoublesP1)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    auto seqs = game.legalTurnSequences(Player::PLAYER1, 1, 2);
    // expect two sequences: {1->2,2->4} and {1->3,3->4}
    EXPECT_TRUE(containsSequence(seqs, {{1, 2}, {2, 4}}));
    EXPECT_TRUE(containsSequence(seqs, {{1, 3}, {3, 4}}));
}

TEST(LegalTurnSeq, NonDoublesP2)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    auto seqs = game.legalTurnSequences(Player::PLAYER2, 1, 2);
    EXPECT_TRUE(containsSequence(seqs, {{6, 5}, {5, 3}}));
    EXPECT_TRUE(containsSequence(seqs, {{6, 4}, {4, 3}}));
}

TEST(LegalTurnSeq, DoublesP1)
{
    Game game(0);
    Player p1("X", Player::PLAYER1), p2("Y", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    auto seqs = game.legalTurnSequences(Player::PLAYER1, 1, 1);

    // Test that we get a reasonable number of sequences should be > 81
    EXPECT_GT(seqs.size(), 81);
    EXPECT_LT(seqs.size(), 1000); // Some reasonable upper bound

    // Test for a valid sequence - four moves from point 19 to 20
    vector<pair<int, int>> target(4, {19, 20});
    EXPECT_TRUE(std::find(seqs.begin(), seqs.end(), target) != seqs.end());

    // Test that all sequences have exactly 4 moves
    for (const auto &seq : seqs)
    {
        EXPECT_EQ(seq.size(), 4);
    }
}

TEST(LegalTurnSeq, Immutability)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    auto before = game.getGameBoard();
    auto seqs = game.legalTurnSequences(Player::PLAYER1, 1, 2);
    auto after = game.getGameBoard();
    EXPECT_EQ(before, after);
}
TEST(LegalTurnSeq, failing_moves_prior)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    game.gameboard.assign({-8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4});
    auto before = game.getGameBoard();
    auto seqs = game.legalTurnSequences(Player::PLAYER1, 3, 1);
    auto after = game.getGameBoard();
    EXPECT_TRUE(seqs.size() != 0);
}
TEST(LegalTurnSeq, trying_replicate)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    game.gameboard.assign({-8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 4});
    string romanos;
    EXPECT_TRUE(game.tryMove(&p1, 5, 22, 25, romanos));
    cout << romanos;
}
TEST(LegalTurnSeq, trying_replicate_another)
{
    Game game(0);
    Player p1("A", Player::PLAYER1), p2("B", Player::PLAYER2);
    game.setPlayers(&p1, &p2);
    game.gameboard.assign({-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});

    EXPECT_NE(game.legalMoves(0, 6).size(), 0);
    EXPECT_EQ(game.legalMoves(0, 6).size(), 1);
    EXPECT_EQ(game.legalTurnSequences(0, 6, 3).size(), 2);
}

TEST(LegalTurnSeq, Player2FinalBearOff)
{
    // Set up a brand-new game, p1 vs p2
    Game game(0);
    Player p1("White", Player::PLAYER1),
        p2("Black", Player::PLAYER2);
    game.setPlayers(&p1, &p2);

    // Manually put exactly one black checker on point 1,
    // and no other checkers on the board.
    // In our representation, gameboard[0] == -1 means ◯ has one checker at pip 1.
    vector<int> empty(24, 0);
    empty[0] = -1;
    game.gameboard = empty;

    // Since all the other ◯ checkers are already freed,
    // canFreePiece() should allow bearing off whenever die >= 1.

    // 1) A single legal origin for die=3:
    auto m = game.legalMoves(Player::PLAYER2, 3);
    EXPECT_EQ(m.size(), 1);
    EXPECT_EQ(m[0], make_pair(1, 0));
    // origin==1 → dest==0 (off the board)

    // 2) And exactly one turn‐sequence, of length 1
    auto seqs = game.legalTurnSequences(Player::PLAYER2, 3, 2);
    ASSERT_EQ(seqs.size(), 2);
    EXPECT_EQ(seqs[0].size(), 1);
    EXPECT_EQ(seqs[0][0], make_pair(1, 0));
}
TEST(LegalTurnSeq, Player2SingleCheckerCanBearOffWithBothDice)
{
    // Arrange: one Black (player2) checker on pip 1, nothing else on board
    Game g(0);
    Player white("W", Player::PLAYER1),
        black("B", Player::PLAYER2);
    g.setPlayers(&white, &black);

    std::vector<int> board24(24, 0);
    board24[0] = -1; // ◯ on point 1
    g.gameboard = board24;

    // Act: roll 3 and 2
    auto seqs = g.legalTurnSequences(Player::PLAYER2, 3, 2);

    // Assert: exactly two single‐step sequences, one per die‐ordering
    ASSERT_EQ(seqs.size(), 2)
        << "Should get two ways to bear off (use 3 first, or 2 first)";
    for (auto &seq : seqs)
    {
        ASSERT_EQ(seq.size(), 1)
            << "Each sequence is a single origin→dest step";
        EXPECT_EQ(seq[0], std::make_pair(1, 0))
            << "Origin=1, Dest=0 (off the board)";
    }
}

// 1) A single ◯ on pip 1, rolling doubles (1,1) should still give you
//    a one‐step bear‐off sequence – not zero.
TEST(LegalTurnSeq, Player2FinalBearOffDouble1)
{
    Game g(0);
    Player w("W", Player::PLAYER1),
        b("B", Player::PLAYER2);
    g.setPlayers(&w, &b);

    // One black checker on point 1, everything else off-board:
    std::vector<int> board(24, 0);
    board[0] = -1;
    g.gameboard = board;

    // Roll doubles 1,1
    auto seqs = g.legalTurnSequences(Player::PLAYER2, 1, 1);

    // You **must** get at least one legal sequence of length 1: (1→0).
    ASSERT_GT(seqs.size(), 0) << "Expected at least one bear-off sequence on double roll";
    for (auto &seq : seqs)
    {
        EXPECT_EQ(seq.size(), 1);
        EXPECT_EQ(seq[0], std::make_pair(1, 0));
    }
}

// 2) Two ◯ checkers on pips 1 and 2, rolling doubles (2,2) should allow
//    you to bear off the '2' and then the '1' (two steps), or vice versa.
TEST(LegalTurnSeq, Player2TwoCheckersDouble2)
{
    Game g(0);
    Player w("W", Player::PLAYER1),
        b("B", Player::PLAYER2);
    g.setPlayers(&w, &b);

    // Black on points 1 and 2:
    std::vector<int> board(24, 0);
    board[0] = board[1] = -1;
    g.gameboard = board;

    // Roll doubles 2,2
    auto seqs = g.legalTurnSequences(Player::PLAYER2, 2, 2);

    // You **must** get at least one two-step sequence:
    // either [(2→0),(1→0)] or [(1→0),(2→0)].
    bool saw_valid = false;
    for (auto &seq : seqs)
    {
        if (seq.size() == 2 &&
            ((seq[0] == std::make_pair(2, 0) && seq[1] == std::make_pair(1, 0)) ||
             (seq[0] == std::make_pair(1, 0) && seq[1] == std::make_pair(2, 0))))
        {
            saw_valid = true;
        }
    }
    EXPECT_TRUE(saw_valid) << "Expected a 2-step double-bear-off sequence for pips {1,2}";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
