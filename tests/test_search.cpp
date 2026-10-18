#include "doctest.h"

#include <cmath>

#include "core/ids.hpp"
#include "eval/cube.hpp"
#include "search/mcsearch.hpp"
#include "search/rollout.hpp"
#include "search/search.hpp"

using namespace bg;

namespace {
const Evaluator& test_evaluator() {
  static Evaluator ev = [] {
    Evaluator e;
    e.init("", std::string(BG_SOURCE_DIR) + "/data");
    Network n;
    n.init(64, 32, 32, 7);
    e.set_network(n);
    return e;
  }();
  return ev;
}
}  // namespace

TEST_CASE("1-ply cube analysis equals the static cube model") {
  const Evaluator& ev = test_evaluator();
  Searcher s(ev, 2, 16);
  Position p;
  REQUIRE(parse_position_id("4HPwATDgc/ABMA", p.board));
  for (int len : {0, 7}) {
    MatchState ms;
    ms.length = len;
    if (len) { ms.score[0] = 2; ms.score[1] = 4; }
    CubeResult r = s.analyze_cube(p.board, ms, SearchConfig::for_plies(1));
    CubeAnalysis a = analyze_cube(ev.evaluate(p.board), ms, cube_efficiency(p.board));
    CHECK(r.analysis.nd == doctest::Approx(a.nd));
    CHECK(r.analysis.dt == doctest::Approx(a.dt));
    CHECK(r.analysis.decision == a.decision);
  }
}

TEST_CASE("move analysis is consistent across depths") {
  const Evaluator& ev = test_evaluator();
  Searcher s(ev, 4, 32);
  Board b = Board::start();
  MatchState ms;
  for (int d = 1; d <= 3; ++d) {
    MoveAnalysis a = s.analyze_moves(b, 3, 1, ms, SearchConfig::for_plies(d));
    REQUIRE(a.moves.size() == 16);
    CHECK(a.moves[0].plies == d);
    for (size_t i = 1; i < a.moves.size(); ++i)
      if (a.moves[i].plies == a.moves[0].plies) CHECK(a.moves[i].value <= a.moves[0].value + 1e-6f);
    for (const auto& m : a.moves) {
      CHECK(m.probs[kWin] >= 0.f);
      CHECK(m.probs[kWin] <= 1.f);
      CHECK(std::isfinite(m.value));
    }
  }
}

TEST_CASE("sampled deep search stays close to full expansion") {
  const Evaluator& ev = test_evaluator();
  Searcher s(ev, 4, 32);
  Board b = Board::start();
  MatchState ms;
  SearchConfig full = SearchConfig::for_plies(4);
  SearchConfig samp = full;
  samp.full_layers = 1;
  samp.sampled_rolls = 7;
  Probs pf = s.evaluate(b, ms, 4);
  s.clear_cache();
  CubeResult rf = s.analyze_cube(b, ms, full);
  s.clear_cache();
  CubeResult rs = s.analyze_cube(b, ms, samp);
  CHECK(std::fabs(rf.analysis.nd - rs.analysis.nd) < 0.05f);
  CHECK(pf[kWin] > 0.f);
}

TEST_CASE("rollout of a trivial bearoff") {
  const Evaluator& ev = test_evaluator();
  ThreadPool pool(2);
  Board b;
  b.set(0, 0, 1);   // one checker on our ace point: we always win
  b.set(1, 0, 15);
  MatchState ms;
  RolloutConfig rc;
  rc.trials = 72;
  RolloutResult r = rollout(ev, b, ms, rc, pool);
  CHECK(r.probs[kWin] == doctest::Approx(1.0));
  CHECK(r.probs[kWinG] == doctest::Approx(1.0));
}

TEST_CASE("MC search: time budget, rollouts and consistent cube values") {
  const Evaluator& ev = test_evaluator();
  Searcher s(ev, 2, 16);
  MatchState ms;
  McConfig mc;
  mc.time_ms = 150;
  mc.window = 1.0f;  // random test network: force several candidates
  McStats st;
  MoveAnalysis a = mc_analyze_moves(s, ev, Board::start(), 6, 4, ms, mc, &st);
  REQUIRE(a.moves.size() == 15);
  CHECK(st.candidates >= 2);
  CHECK(st.games > 0);
  CHECK(st.seconds < 0.5);
  CHECK(a.moves[0].rollouts > 0);
  for (size_t i = 1; i < a.moves.size(); ++i)
    if (a.moves[i].rollouts > 0) CHECK(a.moves[i].value <= a.moves[0].value + 1e-6f);
  mc.cube_window = 10.f;  // always roll out
  CubeResult c = mc_analyze_cube(s, ev, Board::start(), ms, mc, &st);
  CHECK(st.used_rollouts);
  CHECK(std::isfinite(c.analysis.nd));
  CHECK(std::isfinite(c.analysis.dt));
}
