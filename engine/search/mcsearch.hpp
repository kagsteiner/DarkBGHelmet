// Time-controlled rollout search ("MC search"), an alternative to deep expectimax.
//
// Checker play:
//   1. Rank all plays with a shallow search (prefilter_plies) and keep the
//      candidates within `window` of the best (at most max_candidates).
//   2. Each candidate is a bandit arm. Its prior is the shallow value, weighted
//      like `prior_games` rollout games.
//   3. Until the time budget is used up, pick an arm by Thompson sampling
//      (sample each arm's posterior, take the maximum) and roll out `batch`
//      more games of it, truncated after `truncate` plies with a cubeful
//      static evaluation.
//   4. All arms use the same dice for game t (common random numbers), so most
//      of the luck cancels when plays are compared.
//   5. The play with the best posterior mean wins.
// Cube decisions work the same way with the arms "no double" and
// "double/take", but only when the shallow analysis says the decision is close.
#pragma once

#include <cstdint>

#include "core/match.hpp"
#include "search/rollout.hpp"
#include "search/search.hpp"

namespace bg {

struct McConfig {
  int time_ms = 1000;          // budget for a checker decision (including the pre-ranking)
  int cube_time_ms = -1;       // budget for a close cube decision (-1 = time_ms / 2)
  int truncate = 7;            // plies rolled out before the static cubeful evaluation (0 = play to the end)
  int batch = 36;              // games per sampling step (multiples of 36 keep the first roll stratified)
  int prefilter_plies = 2;     // depth of the pre-ranking
  int max_candidates = 8;
  float window = 0.20f;        // candidates within this EMG distance of the best
  float prior_games = 36.f;    // weight of the pre-ranking value, in rollout games
  float cube_window = 0.25f;   // roll out a cube decision only if it is within this EMG margin
  bool variance_reduction = false;
  // Cube actions inside the rollouts (false: cube frozen, cubeful value at the
  // truncation point only; much lower variance).
  bool rollout_cube_actions = false;
  // Paired comparison: candidates are judged by their per-game difference to the
  // current leader on identical dice (much smaller spread than single games).
  // prior_games then weighs the pre-ranking difference, in paired games.
  bool paired = true;
  uint64_t seed = 1;
};

struct McStats {
  int candidates = 0;
  long games = 0;
  int batches = 0;
  double seconds = 0;
  bool used_rollouts = false;  // false: the pre-ranking was decisive
};

MoveAnalysis mc_analyze_moves(Searcher& s, const Evaluator& ev, const Board& b, int d1, int d2, const MatchState& ms,
                              const McConfig& cfg, McStats* stats = nullptr);

CubeResult mc_analyze_cube(Searcher& s, const Evaluator& ev, const Board& b, const MatchState& ms, const McConfig& cfg,
                           McStats* stats = nullptr);

}  // namespace bg
