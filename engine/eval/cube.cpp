#include "cube.hpp"

#include <algorithm>
#include <cmath>

namespace bg {

namespace {
#include "met_kazaross_xg2.inc"
}

float met_mwc(int a, int b, bool post_crawford) {
  if (a <= 0) return 1.f;
  if (b <= 0) return 0.f;
  a = std::min(a, kMaxMatch);
  b = std::min(b, kMaxMatch);
  if (post_crawford && (a == 1 || b == 1)) {
    if (a == 1 && b == 1) return 0.5f;
    return a == 1 ? 1.f - kPost[b - 1] : kPost[a - 1];
  }
  return kPre[a - 1][b - 1];
}

CubeModel::CubeModel(const MatchState& ms) : ms_(ms) {
  // Beyond this cube level we treat the cube as dead (money: 2^8 times the current cube).
  max_cube_ = ms.money() ? ms.cube << 8 : (1 << 12);
}

float CubeModel::value(int pts) const {
  if (ms_.money()) return float(pts) / float(ms_.cube);
  int a = ms_.away(0), b = ms_.away(1);
  if (pts > 0) a -= pts;
  else b += pts;
  if (a <= 0) return 1.f;
  if (b <= 0) return 0.f;
  // Next game is post-Crawford if this one is the Crawford game or already
  // post-Crawford; otherwise someone reaching 1-away starts the Crawford game.
  bool post = ms_.crawford || ms_.post_crawford();
  return met_mwc(a, b, post);
}

bool CubeModel::can_double(int player, int cube) const {
  if (ms_.crawford) return false;
  if (cube >= max_cube_) return false;
  if (!ms_.money() && ms_.away(player) <= cube) return false;
  return true;
}

float CubeModel::avg_win(const Probs& p, int n, bool jacoby) const {
  if (jacoby) return value(n);
  float w = p[kWin];
  if (w <= 1e-7f) return value(n);
  float s = (w - p[kWinG]) * value(n) + (p[kWinG] - p[kWinBG]) * value(2 * n) + p[kWinBG] * value(3 * n);
  return s / w;
}

float CubeModel::avg_lose(const Probs& p, int n, bool jacoby) const {
  if (jacoby) return value(-n);
  float l = 1.f - p[kWin];
  if (l <= 1e-7f) return value(-n);
  float s = (l - p[kLoseG]) * value(-n) + (p[kLoseG] - p[kLoseBG]) * value(-2 * n) + p[kLoseBG] * value(-3 * n);
  return s / l;
}

float CubeModel::dead(const Probs& p, int cube, bool jacoby) const {
  return p[kWin] * avg_win(p, cube, jacoby) + (1.f - p[kWin]) * avg_lose(p, cube, jacoby);
}

// Opponent's take point in my P(win) when I double from n to 2n.
float CubeModel::opp_take_point(const Probs& p, int n) const {
  const int m = 2 * n;
  const float L = avg_lose(p, m, false), W = avg_win(p, m, false);
  const float pass = value(n);
  if (can_double(1, m)) {
    // Opponent owns 2n and redoubles at my take point for 4n, where I pass (lose 2n).
    const float tp4 = my_take_point(p, m);
    const float lo = value(-m);
    if (W - lo <= 1e-9f) return 1.f;
    return std::clamp(tp4 + (1.f - tp4) * (pass - lo) / (W - lo), 0.f, 1.f);
  }
  if (W - L <= 1e-9f) return 1.f;
  return std::clamp((pass - L) / (W - L), 0.f, 1.f);
}

// My take point in my P(win) when the opponent doubles from n to 2n.
float CubeModel::my_take_point(const Probs& p, int n) const {
  const int m = 2 * n;
  const float L = avg_lose(p, m, false), W = avg_win(p, m, false);
  const float pass = value(-n);
  if (can_double(0, m)) {
    // I own 2n and redouble at the opponent's take point for 4n, where it passes.
    const float cp = opp_take_point(p, m);
    const float hi = value(m);
    if (hi - L <= 1e-9f) return 0.f;
    return std::clamp(cp * (pass - L) / (hi - L), 0.f, 1.f);
  }
  if (W - L <= 1e-9f) return 0.f;
  return std::clamp((pass - L) / (W - L), 0.f, 1.f);
}

float CubeModel::cubeful(const Probs& p, int cube, CubeOwner owner, float x) const {
  const bool jacoby = ms_.money() && ms_.jacoby && owner == CubeOwner::Center && cube == 1;
  const float d = dead(p, cube, jacoby);
  const bool me_can = owner != CubeOwner::Opp && can_double(0, cube);
  const bool opp_can = owner != CubeOwner::Me && can_double(1, cube);
  if (!me_can && !opp_can) return d;
  // Piecewise linear live-cube value through (0, L), [(TP, lose cube)], [(CP, win cube)], (1, W).
  float px[4], py[4];
  int k = 0;
  px[k] = 0.f; py[k++] = avg_lose(p, cube, jacoby);
  if (opp_can) { px[k] = my_take_point(p, cube); py[k++] = value(-cube); }
  if (me_can) { px[k] = std::max(opp_take_point(p, cube), k > 1 ? px[k - 1] : 0.f); py[k++] = value(cube); }
  px[k] = 1.f; py[k++] = avg_win(p, cube, jacoby);
  const float w = p[kWin];
  float live = py[k - 1];
  for (int i = 0; i + 1 < k; ++i) {
    if (w <= px[i + 1]) {
      float span = px[i + 1] - px[i];
      live = span > 1e-9f ? py[i] + (py[i + 1] - py[i]) * (w - px[i]) / span : py[i + 1];
      break;
    }
  }
  return x * live + (1.f - x) * d;
}

float CubeModel::to_emg(float v) const {
  if (ms_.money()) return v;
  const float lo = value(-ms_.cube), hi = value(ms_.cube);
  if (hi - lo <= 1e-9f) return 0.f;
  return 2.f * (v - lo) / (hi - lo) - 1.f;
}

float cube_efficiency(const Board& b) {
  // 0.76 reproduces gnubg 1.08's cubeful 0-ply values (measured against it on
  // money and match positions); older gnubg sources used 0.68.
  if (b.contact()) return 0.76f;
  // gnubg's race formula: grows with the pip count of the player on roll.
  return std::clamp(0.55f + 0.00125f * float(b.pips(0)), 0.6f, 0.7f);
}

const char* to_string(CubeDecision d, bool redouble) {
  switch (d) {
    case CubeDecision::NoDouble: return redouble ? "No redouble, take" : "No double, take";
    case CubeDecision::NoDoubleBeaver: return redouble ? "No redouble, beaver" : "No double, beaver";
    case CubeDecision::DoubleTake: return redouble ? "Redouble, take" : "Double, take";
    case CubeDecision::DoublePass: return redouble ? "Redouble, pass" : "Double, pass";
    case CubeDecision::TooGoodTake: return redouble ? "Too good to redouble, take" : "Too good to double, take";
    case CubeDecision::TooGoodPass: return redouble ? "Too good to redouble, pass" : "Too good to double, pass";
    case CubeDecision::NotAvailable: return "Cube not available";
  }
  return "";
}

float CubeAnalysis::best() const {
  switch (decision) {
    case CubeDecision::DoubleTake: return dt;
    case CubeDecision::DoublePass: return dp;
    default: return nd;
  }
}

CubeAnalysis analyze_cube(const Probs& p, const MatchState& ms, float x) {
  CubeModel cm(ms);
  CubeAnalysis a;
  a.probs = p;
  a.x = x;
  const bool jacoby = ms.money() && ms.jacoby && ms.owner == CubeOwner::Center && ms.cube == 1;
  a.cubeless = cm.dead(p, ms.cube, jacoby);
  a.nd = cm.cubeful(p, ms.cube, ms.owner, x);
  a.can_double = ms.owner != CubeOwner::Opp && cm.can_double(0, ms.cube);
  a.dt = cm.cubeful(p, 2 * ms.cube, CubeOwner::Opp, x);
  a.dp = cm.value(ms.cube);
  a.take = a.dt <= a.dp;
  a.beaver = ms.money() && ms.beavers && a.take && a.dt < 0.f;
  if (!a.can_double) {
    a.decision = CubeDecision::NotAvailable;
    return a;
  }
  const float doubled = std::min(a.dt, a.dp);
  if (doubled > a.nd) {
    a.decision = a.take ? CubeDecision::DoubleTake : CubeDecision::DoublePass;
  } else if (a.nd > a.dp) {
    a.decision = a.take ? CubeDecision::TooGoodTake : CubeDecision::TooGoodPass;
  } else {
    a.decision = a.beaver ? CubeDecision::NoDoubleBeaver : CubeDecision::NoDouble;
  }
  return a;
}

}  // namespace bg
