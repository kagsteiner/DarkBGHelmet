// Monte Carlo rollouts with variance reduction.
//
// Each trial plays the game out (or to a truncation depth) with 1-ply checker
// play and, in cubeful mode, static cube decisions. The first two rolls are
// stratified over all 36 x 36 combinations. Luck is removed per turn (Dahl's
// variance reduction): the value of the actual roll minus the average value over
// all 21 rolls, both from the static evaluation, is subtracted from the result.
#pragma once

#include <cstdint>

#include "core/match.hpp"
#include "eval/cube.hpp"
#include "eval/evaluator.hpp"
#include "search/threadpool.hpp"

namespace bg {

struct RolloutConfig {
  int trials = 1296;
  int truncate = 0;          // plies after which the static evaluation is used (0 = play to the end)
  bool variance_reduction = false;  // ~16x lower variance but ~21x the cost per trial
  bool cubeful = true;
  uint64_t seed = 1;
};

struct RolloutResult {
  Probs probs;               // cubeless, view of the player to move at the root
  float cubeless = 0;        // cubeless value (money equity or MWC) at the root cube
  float cubeful = 0;         // cubeful value, the root player does not double first
  float se_cubeless = 0, se_cubeful = 0;  // standard errors
  int trials = 0;
  double seconds = 0;
};

// Rolls out the position with side 0 to move (before its roll). The root player
// does not get to double before its first roll, so `cubeful` is a "no double" value.
RolloutResult rollout(const Evaluator& ev, const Board& b, const MatchState& ms, const RolloutConfig& cfg,
                      ThreadPool& pool);

}  // namespace bg
