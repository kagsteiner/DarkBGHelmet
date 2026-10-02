// Match / cube state, always relative to the player on roll (index 0).
#pragma once

#include <cstdint>
#include <string>

#include "board.hpp"

namespace bg {

enum class CubeOwner : int8_t { Center = 0, Me = 1, Opp = 2 };

constexpr int kMaxMatch = 25;

struct MatchState {
  int length = 0;          // 0 = money game, otherwise 1..25
  int score[2] = {0, 0};   // points already won; [0] = player on roll
  int cube = 1;            // cube value 1, 2, 4, ...
  CubeOwner owner = CubeOwner::Center;
  bool crawford = false;   // this game is the Crawford game
  bool jacoby = true;      // money only
  bool beavers = true;     // money only

  bool money() const { return length == 0; }
  int away(int p) const { return length - score[p]; }

  bool post_crawford() const {
    return !money() && !crawford && (away(0) == 1 || away(1) == 1);
  }

  // May player p (0 = on roll) turn the cube now?
  bool may_double(int p) const {
    if (crawford) return false;
    if (owner == (p == 0 ? CubeOwner::Opp : CubeOwner::Me)) return false;
    if (!money() && away(p) <= cube) return false;  // dead cube: doubling gains nothing
    return true;
  }

  MatchState swapped() const {
    MatchState m = *this;
    std::swap(m.score[0], m.score[1]);
    if (owner == CubeOwner::Me) m.owner = CubeOwner::Opp;
    else if (owner == CubeOwner::Opp) m.owner = CubeOwner::Me;
    return m;
  }

  bool valid(std::string* why = nullptr) const;
};

inline int cube_log2(int cube) {
  int l = 0;
  while ((1 << l) < cube) ++l;
  return l;
}

struct Position {
  Board board;
  MatchState match;
  int dice[2] = {0, 0};  // 0,0 = not yet rolled
};

}  // namespace bg
