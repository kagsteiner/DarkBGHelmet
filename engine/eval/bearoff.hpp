// One-sided bearoff database: up to 15 checkers on the six home points.
//
// For each of the C(21,6) = 54,264 home-board positions we store the exact
// distribution of the number of rolls needed to bear off all checkers (playing
// to minimise the expected number of rolls) and, for positions with all 15
// checkers still on the board, the distribution of rolls needed to bear off the
// first checker (needed for gammon chances). Two-sided bearoff probabilities are
// derived from the two independent distributions.
#pragma once

#include <string>
#include <vector>

#include "core/board.hpp"
#include "eval/probs.hpp"

namespace bg {

class BearoffDB {
 public:
  static constexpr int kPositions = 54264;
  static constexpr int kMaxRolls = 32;

  // Loads from `path` if it exists, otherwise computes and saves it there.
  bool init(const std::string& path);
  bool ready() const { return !off_.empty(); }

  static bool in_db(u128 side) { return (side & kOutsideHome) == 0; }
  static int index(u128 side);

  // Both sides must be in_db and the game not over. Board side 0 is on roll.
  Probs evaluate(const Board& b) const;

  const float* off_dist(int idx) const { return &off_[size_t(idx) * kMaxRolls]; }
  const float* first_dist(int idx) const { return &first_[size_t(idx) * kMaxRolls]; }
  float mean_rolls(int idx) const;

 private:
  void compute();
  std::vector<float> off_, first_;
};

BearoffDB& bearoff_db();  // process-wide instance

}  // namespace bg
