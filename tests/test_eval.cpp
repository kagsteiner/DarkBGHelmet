#include "doctest.h"

#include <cmath>
#include <random>

#include "core/movegen.hpp"
#include "eval/bearoff.hpp"
#include "eval/evaluator.hpp"
#include "eval/features.hpp"
#include "nn/network.hpp"

using namespace bg;

namespace {
std::vector<Board> random_positions(int n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::vector<Board> out;
  std::vector<Move> moves;
  while ((int)out.size() < n) {
    Board b = Board::start();
    int plies = int(rng() % 90);
    for (int i = 0; i < plies && b.winner() < 0; ++i) {
      generate_moves(b, int(rng() % 6) + 1, int(rng() % 6) + 1, moves);
      b = moves[rng() % moves.size()].after.swapped();
    }
    if (b.winner() < 0) out.push_back(b);
  }
  return out;
}
}  // namespace

TEST_CASE("incremental accumulator equals refresh") {
  Network net;
  net.init(128, 32, 32, 3);
  std::vector<Move> moves;
  std::mt19937 rng(5);
  for (const Board& b : random_positions(300, 11)) {
    Accumulator pa, ca, ra;
    acc_refresh(net, b, pa);
    generate_moves(b, int(rng() % 6) + 1, int(rng() % 6) + 1, moves);
    for (const Move& m : moves) {
      acc_update(net, b, pa, m.after, ca);
      acc_refresh(net, m.after, ra);
      float maxd = 0;
      for (int p = 0; p < 2; ++p)
        for (int i = 0; i < net.H; ++i) maxd = std::max(maxd, std::fabs(ca.v[p][i] - ra.v[p][i]));
      CHECK(maxd < 1e-4f);
    }
  }
}

TEST_CASE("hit chance agrees with brute force") {
  std::vector<Move> moves;
  int agree = 0, total = 0;
  for (const Board& b : random_positions(1500, 21)) {
    int hits = 0;
    for (int d1 = 1; d1 <= 6; ++d1)
      for (int d2 = d1; d2 <= 6; ++d2) {
        generate_moves(b, d1, d2, moves);
        bool h = false;
        for (const Move& m : moves) h |= m.hits != 0;
        if (h) hits += d1 == d2 ? 1 : 2;
      }
    ++total;
    if (std::fabs(hits / 36.f - hit_chance(b)) < 1e-6f) ++agree;
  }
  MESSAGE("hit chance exact on " << agree << " / " << total);
  CHECK(agree >= total * 97 / 100);
}

TEST_CASE("bearoff database") {
  BearoffDB& db = bearoff_db();
  db.init(std::string(BG_SOURCE_DIR) + "/data/bearoff_os15.bin");
  Board b;
  b.set(0, 5, 1);  // one checker on our 6 point
  b.set(1, 0, 1);  // opponent: one checker on its ace point
  Probs p = db.evaluate(b);
  CHECK(p[kWin] == doctest::Approx(27.0 / 36.0).epsilon(1e-5));
  // 15 checkers on the ace point need 8 rolls without doubles.
  Board c;
  c.set(0, 0, 15);
  int idx = BearoffDB::index(c.side[0]);
  CHECK(db.mean_rolls(idx) > 6.5f);
  CHECK(db.mean_rolls(idx) < 8.0f);
  CHECK(BearoffDB::index(0) == 0);
}

TEST_CASE("terminal probabilities") {
  Board b;
  b.set(1, 20, 15);  // side 0 has no checkers: side 0 won; side 1 never bore off and is in our home
  Probs p = Evaluator::terminal(b);
  CHECK(p[kWin] == 1.f);
  CHECK(p[kWinG] == 1.f);
  CHECK(p[kWinBG] == 1.f);
}
