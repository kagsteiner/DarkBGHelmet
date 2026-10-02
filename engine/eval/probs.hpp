// Cubeless game-outcome probabilities, gnubg/XG convention.
#pragma once

#include <algorithm>
#include <array>

namespace bg {

enum { kWin = 0, kWinG = 1, kWinBG = 2, kLoseG = 3, kLoseBG = 4 };

struct Probs {
  // Unconditional: P(win), P(win gammon or better), P(win backgammon),
  // P(lose gammon or worse), P(lose backgammon). Player on roll's view.
  std::array<float, 5> p{0.5f, 0.f, 0.f, 0.f, 0.f};

  float operator[](int i) const { return p[i]; }
  float& operator[](int i) { return p[i]; }

  // Cubeless money equity.
  float equity() const { return 2.f * p[kWin] - 1.f + p[kWinG] - p[kLoseG] + p[kWinBG] - p[kLoseBG]; }

  // The same outcome seen by the other player.
  Probs inverted() const {
    Probs q;
    q.p = {1.f - p[kWin], p[kLoseG], p[kLoseBG], p[kWinG], p[kWinBG]};
    return q;
  }

  // Enforce 1 >= win >= wg >= wbg and (1-win) >= lg >= lbg.
  void sanitize() {
    for (float& x : p) x = std::clamp(x, 0.f, 1.f);
    p[kWinG] = std::min(p[kWinG], p[kWin]);
    p[kWinBG] = std::min(p[kWinBG], p[kWinG]);
    p[kLoseG] = std::min(p[kLoseG], 1.f - p[kWin]);
    p[kLoseBG] = std::min(p[kLoseBG], p[kLoseG]);
  }

  Probs& operator+=(const Probs& o) { for (int i = 0; i < 5; ++i) p[i] += o.p[i]; return *this; }
  Probs operator*(float w) const { Probs q; for (int i = 0; i < 5; ++i) q.p[i] = p[i] * w; return q; }
  static Probs zero() { Probs q; q.p = {0, 0, 0, 0, 0}; return q; }
};

}  // namespace bg
