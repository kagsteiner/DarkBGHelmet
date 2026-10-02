#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <random>

#include "core/board.hpp"
#include "core/ids.hpp"
#include "core/movegen.hpp"

using namespace bg;

TEST_CASE("start position basics") {
  Board b = Board::start();
  CHECK(b.on_board(0) == 15);
  CHECK(b.on_board(1) == 15);
  CHECK(b.pips(0) == 167);
  CHECK(b.contact());
  CHECK(b.valid());
  CHECK(position_id(b) == "4HPwATDgc/ABMA");
}

TEST_CASE("position id round trip on random positions") {
  std::mt19937_64 rng(7);
  std::vector<Move> moves;
  for (int g = 0; g < 200; ++g) {
    Board b = Board::start();
    for (int ply = 0; ply < 60 && b.winner() < 0; ++ply) {
      int d1 = int(rng() % 6) + 1, d2 = int(rng() % 6) + 1;
      generate_moves(b, d1, d2, moves);
      b = moves[rng() % moves.size()].after.swapped();
      CHECK(b.valid());
      CHECK(b.on_board(0) + b.off(0) == 15);
      Board r;
      REQUIRE(parse_position_id(position_id(b), r));
      CHECK(r == b);
      Position p;
      p.board = b;
      p.match.length = 7;
      p.match.score[0] = 2;
      p.match.score[1] = 5;
      p.match.cube = 2;
      p.match.owner = CubeOwner::Opp;
      Position q;
      REQUIRE(parse_xgid(to_xgid(p), q));
      CHECK(q.board == b);
      CHECK(q.match.score[1] == 5);
      CHECK(q.match.owner == CubeOwner::Opp);
    }
  }
}

TEST_CASE("xgid start") {
  Position p;
  REQUIRE(parse_xgid("XGID=-b----E-C---eE---c-e----B-:0:0:1:52:0:0:0:0:10", p));
  CHECK(p.board == Board::start());
  CHECK(p.dice[0] == 5);
  CHECK(p.dice[1] == 2);
  CHECK(p.match.money());
}

TEST_CASE("match id round trip") {
  MatchState m;
  m.length = 9;
  m.score[0] = 3;
  m.score[1] = 7;
  m.cube = 4;
  m.owner = CubeOwner::Me;
  MatchState r;
  int d1, d2;
  REQUIRE(parse_match_id(match_id(m, 6, 2), r, d1, d2));
  CHECK(r.length == 9);
  CHECK(r.score[0] == 3);
  CHECK(r.score[1] == 7);
  CHECK(r.cube == 4);
  CHECK(r.owner == CubeOwner::Me);
  CHECK(d1 == 6);
  CHECK(d2 == 2);
}

TEST_CASE("opening moves") {
  Board b = Board::start();
  std::vector<Move> moves;
  generate_moves(b, 3, 1, moves);
  int idx = find_move(moves, b, "8/5 6/5");
  REQUIRE(idx >= 0);
  CHECK(moves[idx].str() == "8/5 6/5");
  generate_moves(b, 6, 6, moves);
  idx = find_move(moves, b, "24/18(2) 13/7(2)");
  REQUIRE(idx >= 0);
  CHECK(moves[idx].str() == "24/18(2) 13/7(2)");
}

TEST_CASE("bar entry and blocked") {
  Board b;
  b.set(0, kBar, 1);
  b.set(0, 5, 14);
  // Opponent closes its home board (our slots 18..23 = its slots 0..5).
  for (int t = 0; t < 6; ++t) b.set(1, t, 2);
  b.set(1, 10, 3);
  std::vector<Move> moves;
  generate_moves(b, 6, 5, moves);
  REQUIRE(moves.size() == 1);
  CHECK(moves[0].n == 0);
}

TEST_CASE("bear off larger die rule") {
  Board b;
  b.set(0, 2, 1);  // a checker on the 3 point
  b.set(1, 20, 15);
  std::vector<Move> moves;
  generate_moves(b, 6, 1, moves);
  // Either 3/off (6) or 3/2/off... 3/2 then 2/off with 6 works; both bear off: one result.
  REQUIRE(moves.size() == 1);
  CHECK(moves[0].after.side[0] == 0);
}
