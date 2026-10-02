#include "doctest.h"

#include "eval/cube.hpp"

using namespace bg;

namespace {
Probs gammonless(float w) {
  Probs p = Probs::zero();
  p[kWin] = w;
  return p;
}
}  // namespace

TEST_CASE("MET basics") {
  CHECK(met_mwc(1, 1, false) == doctest::Approx(0.5));
  CHECK(met_mwc(1, 2, false) == doctest::Approx(0.67736));
  CHECK(met_mwc(2, 1, true) == doctest::Approx(0.48803));
  CHECK(met_mwc(1, 2, true) == doctest::Approx(1 - 0.48803));
  CHECK(met_mwc(0, 5, false) == 1.f);
  for (int a = 1; a <= 25; ++a)
    for (int b = 1; b <= 25; ++b) CHECK(met_mwc(a, b, false) + met_mwc(b, a, false) == doctest::Approx(1.0).epsilon(1e-3));
}

TEST_CASE("money take points without gammons") {
  MatchState ms;
  ms.jacoby = false;
  CubeModel cm(ms);
  Probs p = gammonless(0.5f);
  // Fully live cube: take point 20%, so the opponent passes above 80%.
  CHECK(cm.my_take_point(p, 1) == doctest::Approx(0.2).epsilon(0.01));
  CHECK(cm.opp_take_point(p, 1) == doctest::Approx(0.8).epsilon(0.01));
}

TEST_CASE("money cube decisions without gammons") {
  MatchState ms;
  ms.jacoby = false;
  ms.beavers = false;
  auto a = analyze_cube(gammonless(0.50f), ms, 0.68f);
  CHECK(a.decision == CubeDecision::NoDouble);
  a = analyze_cube(gammonless(0.75f), ms, 0.68f);
  CHECK(a.decision == CubeDecision::DoubleTake);
  a = analyze_cube(gammonless(0.85f), ms, 0.68f);
  CHECK(a.decision == CubeDecision::DoublePass);
  CHECK(a.best() == doctest::Approx(1.0));
  // With a fully live cube, doubling at 75% is premature.
  a = analyze_cube(gammonless(0.75f), ms, 1.0f);
  CHECK(a.decision == CubeDecision::NoDouble);
}

TEST_CASE("too good with many gammons") {
  MatchState ms;
  ms.jacoby = false;
  Probs p = Probs::zero();
  p[kWin] = 0.88f;
  p[kWinG] = 0.6f;
  auto a = analyze_cube(p, ms, 0.68f);
  CHECK(a.decision == CubeDecision::TooGoodPass);
}

TEST_CASE("match play cube availability") {
  MatchState ms;
  ms.length = 5;
  ms.score[0] = 4;  // 1-away
  ms.score[1] = 2;  // 3-away
  ms.crawford = true;
  auto a = analyze_cube(gammonless(0.7f), ms, 0.68f);
  CHECK(a.decision == CubeDecision::NotAvailable);
  // DMP: cube dead, value = P(win)
  MatchState dmp;
  dmp.length = 7;
  dmp.score[0] = dmp.score[1] = 6;
  a = analyze_cube(gammonless(0.6f), dmp, 0.68f);
  CHECK(a.decision == CubeDecision::NotAvailable);
  CHECK(a.nd == doctest::Approx(0.6));
}

TEST_CASE("match play 2-away 2-away doubles early") {
  MatchState ms;
  ms.length = 5;
  ms.score[0] = ms.score[1] = 3;
  auto a = analyze_cube(gammonless(0.62f), ms, 0.68f);
  CHECK((a.decision == CubeDecision::DoubleTake || a.decision == CubeDecision::DoublePass));
}
