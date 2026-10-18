// Selective expectimax with sparse chance sampling (SESCS).
//
// Depth uses the XG convention: 1-ply = static evaluation. Evaluating a position
// P (side 0 to roll) at depth d > 1 averages over the 21 rolls; for each roll the
// mover's play is chosen by its static value and the resulting position is
// evaluated at depth d - 1.
//
// Cube handling follows gnubg's recursive scheme: every node is evaluated for a
// small set of cube states at once. At each internal node the player to move
// takes its optimal cube action (no double vs. double/take vs. double/pass) for
// every state; leaves use the generalised Janowski formula.
//
// Speed-ups:
//  * static move choice at inner nodes (branching 21 per ply, not 21 x moves)
//  * incremental NNUE accumulators for all children of a node
//  * at deep chance nodes only k representative rolls are searched; the other
//    rolls are covered by a control variate (the static value of every roll is
//    known anyway, the deep search only corrects it), which keeps the estimate
//    unbiased with respect to the static evaluation and has low variance
//  * transposition table, parallel search over root candidates x first rolls
#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "core/match.hpp"
#include "core/movegen.hpp"
#include "eval/cube.hpp"
#include "eval/evaluator.hpp"
#include "search/threadpool.hpp"

namespace bg {

struct CubeState {
  int16_t cube = 1;
  CubeOwner owner = CubeOwner::Center;  // relative to the player to move at the node
  bool operator==(const CubeState& o) const { return cube == o.cube && owner == o.owner; }
};

constexpr int kMaxStates = 16;

struct SearchConfig {
  int plies = 2;            // XG convention, 1..7
  int full_layers = 3;      // chance layers (from the evaluated node) expanded with all 21 rolls
  int sampled_rolls = 7;    // groups of similar rolls searched at deeper chance layers (21 = no sampling)
  int tail_rolls = 0;       // extreme rolls (jokers / anti-jokers) at each end searched individually
  bool cubeful = true;
  // Root move filter per depth step: keep at most `keep[d]` moves within `width[d]`
  // (EMG) of the best when going from depth d to d+1.
  int keep[8] = {0, 16, 8, 5, 3, 2, 2, 2};
  float width[8] = {0, 0.16f, 0.10f, 0.07f, 0.05f, 0.04f, 0.04f, 0.04f};
  int max_candidates = 0;   // 0 = no extra cap

  // Sensible defaults for a target depth.
  static SearchConfig for_plies(int plies);
};

struct MoveEval {
  Move move;
  Probs probs;        // cubeless, mover's view
  float value = 0;    // cubeful (or cubeless) value, mover's view, value space
  float emg = 0;      // money-equivalent equity
  int plies = 1;      // depth this move was evaluated at
  int rollouts = 0;   // rollout games (MC search only)
};

struct MoveAnalysis {
  std::vector<MoveEval> moves;  // sorted best first; deeper evaluations rank first
  u64 nodes = 0;
  double seconds = 0;
};

struct CubeResult {
  CubeAnalysis analysis;  // values in value space (MWC for matches)
  float nd_emg = 0, dt_emg = 0, dp_emg = 0;
  int plies = 1;
  u64 nodes = 0;
  double seconds = 0;
};

class TranspositionTable;

class Searcher {
 public:
  Searcher(const Evaluator& ev, int threads = 0, size_t tt_mb = 256);
  ~Searcher();

  void set_threads(int threads);
  ThreadPool& pool() { return pool_; }
  int threads() const { return pool_.size(); }
  void clear_cache();

  // Cube decision for the player to move (side 0) before rolling.
  CubeResult analyze_cube(const Board& b, const MatchState& ms, const SearchConfig& cfg);

  // Ranks the legal plays for a roll.
  // `must_include` (optional): the position after a specific play (mover's view)
  // that is always kept among the candidates, so it is evaluated at full depth.
  MoveAnalysis analyze_moves(const Board& b, int d1, int d2, const MatchState& ms, const SearchConfig& cfg,
                             const Board* must_include = nullptr);

  // Cubeless probabilities of a position at depth (side 0 to roll).
  Probs evaluate(const Board& b, const MatchState& ms, int plies);

  // Request to stop the current search as soon as possible.
  void stop() { stop_.store(true); }

 private:
  struct StateSet {
    CubeState s[kMaxStates];
    int n = 0;
    int add(CubeState c);
    u64 hash() const;
  };
  struct Ctx;
  struct Root;
  struct Node;
  struct RollOut;

  void make_node(const Root& R, const Board& P, int parity, const StateSet& S, Node& n) const;
  // Values for a fixed cube state set: the player to move does not act on the
  // cube at this node (its options at later nodes are included).
  void eval_fixed(const Root& R, const Board& P, int d, int parity, const StateSet& S, Probs& out_p, float* out_v,
                  Ctx& ctx, int layer);
  void eval_nodes_parallel(const Root& R, const std::vector<Board>& boards, int d, int parity,
                           const std::vector<StateSet>& states, std::vector<Probs>& out_p,
                           std::vector<std::array<float, kMaxStates>>& out_v);
  void static_roll(const Root& R, const Node& n, int r, RollOut& o, Ctx& ctx, int layer) const;
  void deep_roll(const Root& R, const Node& n, const RollOut& o, int d, Probs& dp, float* dv, Ctx& ctx, int layer);
  void leaf_values(const Root& R, const Board& P, const Probs& p, int parity, const StateSet& S, float* v) const;
  // The player to move at `cp` picks its best cube action given fixed values.
  float child_optimal(const Root& R, int cp, const float* v, int nd, int dt, int cube) const;
  void terminal_values(const Root& R, const Board& after, int parity, const StateSet& S, Probs& p, float* v) const;

  const Evaluator& ev_;
  ThreadPool pool_;
  std::unique_ptr<TranspositionTable> tt_;
  std::atomic<bool> stop_{false};
  std::atomic<u64> nodes_{0};
};

}  // namespace bg
